import hashlib
import json
import math
import os
import queue
import subprocess
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


class Stage:
    def __init__(self, spec):
        self.name = spec["name"]
        self.sims = int(spec["sims"])
        self.garbage_scale = float(spec.get("garbage_scale", 1.0))
        self.horizon = int(spec.get("horizon", 100))
        if spec.get("env", "hmm") != "hmm":
            raise ValueError(f"stage {self.name}: env {spec['env']!r}: only the hmm opponent is left")
        self.rules = spec.get("rules", "tetrio")
        self.objective = spec.get("objective", "strength")
        if self.objective not in ("strength", "sims-per-second", "game-score"):
            raise ValueError(f"unknown objective: {self.objective}")
        self.spec = dict(spec)

    def args(self):
        return ["--sims", str(self.sims), "--garbage-scale", repr(self.garbage_scale),
                "--max-pieces", str(self.horizon)]

    def config(self, config):
        return dict(config, rules=self.rules)


def fitness(game, horizon, objective="strength"):
    if objective == "sims-per-second":
        sims = game.get("sims_per_move", 0)
        ms = game.get("ms_per_move", 0)
        if not (math.isfinite(sims) and sims > 0 and math.isfinite(ms) and ms > 0):
            raise RunnerError("sims-per-second needs positive, finite sims_per_move and ms_per_move from nana_emulate")
        return 1000.0 * sims / ms
    if objective == "game-score":
        score = game.get("game_score")
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise RunnerError("game-score needs a finite game_score from nana_emulate")
        return score
    if objective != "strength":
        raise ValueError(f"unknown objective: {objective}")
    return game["attack"] / horizon


class RunnerError(RuntimeError):
    pass


class Runner:
    @staticmethod
    def validate_cpus(cpus, threads):
        if not isinstance(threads, int) or not 1 <= threads <= 64:
            raise ValueError("threads must be an integer in [1, 64]")
        if not cpus or any(not isinstance(cpu, int) or cpu < 0 for cpu in cpus):
            raise ValueError("cpus must be a nonempty list of nonnegative CPU numbers")
        if len(set(cpus)) != len(cpus):
            raise ValueError("cpus must not contain duplicates")
        if len(cpus) % threads:
            raise ValueError("--threads must divide the number of --cpus (each game needs its own CPU group)")

    def __init__(self, nana, cpus, workdir, chunk=4, timeout=3600, threads=1):
        self.nana = os.path.abspath(nana)
        self.cpus = list(cpus)
        self.validate_cpus(self.cpus, threads)
        self.threads = threads
        self.cpu_groups = [self.cpus[i:i + threads] for i in range(0, len(self.cpus), threads)]
        self.workers = len(self.cpu_groups)
        self.config_dir = os.path.join(workdir, "configs")
        os.makedirs(self.config_dir, exist_ok=True)
        self.chunk = chunk
        self.timeout = timeout
        self.free = queue.Queue()
        for group in self.cpu_groups:
            self.free.put(group)
        self.games_played = 0
        self.failures = 0
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.procs = set()
        self.progress = None
        self.chunk_times = []

    def config_path(self, config):
        text = json.dumps(config, sort_keys=True)
        name = hashlib.sha1(text.encode()).hexdigest()[:16] + ".json"
        path = os.path.join(self.config_dir, name)
        if not os.path.exists(path):
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                f.write(text)
            os.replace(tmp, path)
        return path

    def _env(self, config_path):
        env = dict(os.environ)
        env["NANA_CONFIG"] = config_path
        return env

    def print_config(self, config):
        out = subprocess.run([self.nana, "--print-config"], env=self._env(self.config_path(config)),
                             capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            raise RunnerError(f"nana_emulate --print-config failed: {out.stderr.strip()}")
        return json.loads(out.stdout)

    def _run_chunk(self, config_path, stage, seed0, games):
        cpus = self.free.get()
        try:
            affinity = ["--cpu", str(cpus[0])] if self.threads == 1 else ["--cpus", ",".join(map(str, cpus))]
            cmd = [self.nana, "--threads", str(self.threads)] + affinity + ["--search-seed", "1", "--games", str(games),
                   "--seed", str(seed0)] + stage.args()
            for attempt in range(2):
                started = time.monotonic()
                returncode, stdout, stderr = self._spawn(cmd, self._env(config_path))
                results = [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]
                if returncode == 0 and len(results) == games:
                    played = {g.get("rules") for g in results}
                    if played != {stage.rules}:
                        raise RunnerError(f"asked for {stage.rules} rules, nana_emulate played {sorted(map(str, played))}: "
                                          f"{' '.join(cmd)}")
                    with self.lock:
                        self.games_played += games
                        self.chunk_times.append((time.monotonic() - started, games))
                    return results
                with self.lock:
                    self.failures += 1
            raise RunnerError(f"nana_emulate failed twice (exit {returncode}): {' '.join(cmd)}\n{stderr[-2000:]}")
        finally:
            self.free.put(cpus)

    def _spawn(self, cmd, env):
        if self.stopping.is_set():
            raise KeyboardInterrupt
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        with self.lock:
            self.procs.add(proc)
            stopping = self.stopping.is_set()
        if stopping:
            proc.kill()
        try:
            stdout, stderr = proc.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
        finally:
            with self.lock:
                self.procs.discard(proc)
        if self.stopping.is_set():
            raise KeyboardInterrupt
        return proc.returncode, stdout, stderr

    def stop(self):
        with self.lock:
            self.stopping.set()
            procs = list(self.procs)
        for proc in procs:
            try:
                proc.kill()
            except OSError:
                pass

    def play(self, requests):
        tasks = []
        for key, config, stage, seed0, games in requests:
            path = self.config_path(stage.config(config))
            for start in range(0, games, self.chunk):
                n = min(self.chunk, games - start)
                tasks.append((key, start, path, stage, seed0 + start, n))
        results = {key: [None] * games for key, _, _, _, games in requests}
        total, done = sum(games for *_, games in requests), 0
        pool = ThreadPoolExecutor(max_workers=self.workers)
        try:
            pending = {pool.submit(self._run_chunk, path, stage, seed, n): (key, start)
                       for key, start, path, stage, seed, n in tasks}
            while pending:
                finished, _ = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                for future in finished:
                    key, start = pending.pop(future)
                    chunk = future.result()
                    for i, game in enumerate(chunk):
                        results[key][start + i] = game
                    done += len(chunk)
                if self.progress:
                    self.progress(done, total)
        except KeyboardInterrupt:
            self.stop()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown()
        return results

    def scores(self, requests):
        games = self.play(requests)
        stages = {key: stage for key, _, stage, _, _ in requests}
        return {key: [fitness(g, stages[key].horizon, stages[key].objective) for g in gs]
                for key, gs in games.items()}, games
