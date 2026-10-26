import hashlib
import json
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
        self.spec = dict(spec)

    def args(self):
        return ["--sims", str(self.sims), "--garbage-scale", repr(self.garbage_scale),
                "--max-pieces", str(self.horizon)]

    def config(self, config):
        return dict(config, rules=self.rules)


def fitness(game, horizon):
    return game["attack"] / horizon


class RunnerError(RuntimeError):
    pass


class Runner:
    def __init__(self, nana, cpus, workdir, chunk=4, timeout=3600):
        self.nana = os.path.abspath(nana)
        self.cpus = list(cpus)
        self.config_dir = os.path.join(workdir, "configs")
        os.makedirs(self.config_dir, exist_ok=True)
        self.chunk = chunk
        self.timeout = timeout
        self.free = queue.Queue()
        for cpu in self.cpus:
            self.free.put(cpu)
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
        cpu = self.free.get()
        try:
            cmd = [self.nana, "--threads", "1", "--cpu", str(cpu), "--search-seed", "1", "--games", str(games),
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
            self.free.put(cpu)

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
        pool = ThreadPoolExecutor(max_workers=len(self.cpus))
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
        horizons = {key: stage.horizon for key, _, stage, _, _ in requests}
        return {key: [fitness(g, horizons[key]) for g in gs] for key, gs in games.items()}, games
