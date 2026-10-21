"""Plays seeded games of nana_play, one single-thread game process per cpu, and scores them.

Every game is deterministic given (config, stage, seed): nana_play --threads 1 --sims N
--search-seed 1 replays exactly, so configs compared on the same seeds see the same pieces,
garbage and opponent (common random numbers).
"""

import hashlib
import json
import os
import queue
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor


class Stage:
    """How games are played: search budget, opponent pressure, game length and opponent."""

    def __init__(self, spec):
        self.name = spec["name"]
        self.sims = int(spec["sims"])
        self.garbage_scale = float(spec.get("garbage_scale", 1.0))
        self.horizon = int(spec.get("horizon", 100))
        # hmm: the game plays the opponent HMM (MarkovOpponent, fitted on nana's stadium games) and
        # the search models it; handcrafted: the old Opponent for both
        self.env = spec.get("env", "hmm")
        self.spec = dict(spec)

    def args(self):
        return ["--sims", str(self.sims), "--garbage-scale", repr(self.garbage_scale),
                "--max-pieces", str(self.horizon), "--markov-env", "1" if self.env == "hmm" else "0"]

    def environment(self):
        return {"NANA_RULES": "tetrio", "NANA_REWARD": "app",
                "NANA_MARKOV_OPPONENT": "1" if self.env == "hmm" else "0"}


def fitness(game, horizon):
    """Attack per piece over the full horizon: pieces lost to dying early count as no attack, so
    survival is part of the score without a separate penalty."""
    return game["attack"] / horizon


class RunnerError(RuntimeError):
    pass


class Runner:
    def __init__(self, nana, cpus, workdir, chunk=4, timeout=3600):
        self.nana = nana
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

    def config_path(self, config):
        """configs are content-addressed, so a file is written once and never changes"""
        text = json.dumps(config, sort_keys=True)
        name = hashlib.sha1(text.encode()).hexdigest()[:16] + ".json"
        path = os.path.join(self.config_dir, name)
        if not os.path.exists(path):
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                f.write(text)
            os.replace(tmp, path)
        return path

    def _env(self, config_path, stage):
        env = dict(os.environ)
        env.update(stage.environment())
        env["NANA_CONFIG"] = config_path
        return env

    def print_config(self, config):
        """The config nana actually loads for this file (nana_play --print-config)."""
        out = subprocess.run([self.nana, "--print-config"], env=self._env(self.config_path(config), Stage(
            {"name": "check", "sims": 1})), capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            raise RunnerError(f"nana_play --print-config failed: {out.stderr.strip()}")
        return json.loads(out.stdout)

    def _run_chunk(self, config_path, stage, seed0, games):
        cpu = self.free.get()
        try:
            cmd = [self.nana, "--threads", "1", "--cpu", str(cpu), "--search-seed", "1", "--games", str(games),
                   "--seed", str(seed0)] + stage.args()
            for attempt in range(2):
                out = subprocess.run(cmd, env=self._env(config_path, stage), capture_output=True, text=True,
                                     timeout=self.timeout)
                results = [json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")]
                if out.returncode == 0 and len(results) == games:
                    with self.lock:
                        self.games_played += games
                    return results
                with self.lock:
                    self.failures += 1
            raise RunnerError(f"nana_play failed twice (exit {out.returncode}): {' '.join(cmd)}\n{out.stderr[-2000:]}")
        finally:
            self.free.put(cpu)

    def play(self, requests):
        """requests: [(key, config, stage, seed0, games)]. Returns {key: [game result per seed, in seed
        order]}. Games are split into chunks and spread over the cpus."""
        tasks = []
        for key, config, stage, seed0, games in requests:
            path = self.config_path(config)
            for start in range(0, games, self.chunk):
                n = min(self.chunk, games - start)
                tasks.append((key, start, path, stage, seed0 + start, n))
        results = {key: [None] * games for key, _, _, _, games in requests}
        with ThreadPoolExecutor(max_workers=len(self.cpus)) as pool:
            futures = [(key, start, pool.submit(self._run_chunk, path, stage, seed, n))
                       for key, start, path, stage, seed, n in tasks]
            for key, start, future in futures:
                for i, game in enumerate(future.result()):
                    results[key][start + i] = game
        return results

    def scores(self, requests):
        """Like play, but {key: [fitness per seed]} plus the raw games."""
        games = self.play(requests)
        horizons = {key: stage.horizon for key, _, stage, _, _ in requests}
        return {key: [fitness(g, horizons[key]) for g in gs] for key, gs in games.items()}, games
