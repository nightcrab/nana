"""Head-to-head fitness using Universal-Shaktris-Stadium's stadium_cli."""
import json
import math
from contextlib import closing, contextmanager
import os
from pathlib import Path
import shutil
import shlex
import sqlite3
import tempfile
import time
from statistics import NormalDist

from .runner import Runner, RunnerError


def reference_interval(games):
    """95% Wilson interval for a 50% score, assuming independent win/loss games.

    This describes finite-sample uncertainty, not uncertainty in the fixed baseline.
    Draws scored as half a win reduce variance relative to this reference.
    """
    if games < 1:
        raise ValueError("games must be positive")
    z = NormalDist().inv_cdf(0.975)
    half = z / (2 * math.sqrt(games + z * z))
    return 0.5 - half, 0.5 + half


class StadiumRunner(Runner):
    @staticmethod
    def validate_cpus(cpus, threads):
        Runner.validate_cpus(cpus, threads)
        if len(cpus) % (2 * threads):
            raise ValueError("stadium needs two CPU groups per match: CPU count must be divisible by 2 * threads")

    def __init__(self, nana, cpus, workdir, stadium, bot, **kwargs):
        super().__init__(nana, cpus, workdir, **kwargs)
        self.cpu_groups = [self.cpus[i:i + 2 * self.threads]
                           for i in range(0, len(self.cpus), 2 * self.threads)]
        self.workers = len(self.cpu_groups)
        while not self.free.empty():
            self.free.get_nowait()
        for group in self.cpu_groups:
            self.free.put(group)
        self.stadium = os.path.abspath(stadium)
        self.bot = Path(bot).resolve()
        self.baseline_config = None
        self.match_dir = os.path.abspath(os.path.join(workdir, "matches"))
        os.makedirs(self.match_dir, exist_ok=True)

    def _bot_copy(self, directory, config, cpus):
        directory.mkdir()
        target = directory / self.bot.name
        shutil.copy2(self.bot, target)
        # Windows runtime dependencies must be beside the isolated executable.
        for dll in self.bot.parent.glob("*.dll"):
            shutil.copy2(dll, directory / dll.name)
        (directory / "nana.json").write_text(json.dumps(dict(config, threads=self.threads)))
        if os.name != "nt":
            # Both bots otherwise pin their first search worker to the first
            # CPU in the stadium's inherited mask, leaving half the cores idle.
            launcher = directory / "bot.sh"
            launcher.write_text(f"#!/bin/sh\nexec taskset -c {','.join(map(str, cpus))} {shlex.quote(str(target))}\n")
            launcher.chmod(0o700)
            return str(launcher)
        return str(target)

    @contextmanager
    def _match_directory(self):
        with tempfile.TemporaryDirectory(dir=self.match_dir) as temporary:
            root = Path(temporary)
            try:
                yield root
            except Exception as exc:
                failure = Path(self.match_dir).parent / "failed-matches" / root.name
                failure.parent.mkdir(exist_ok=True)
                shutil.copytree(root, failure)
                raise RunnerError(f"{exc}; match artifacts: {failure}") from exc

    def _run_chunk(self, config_path, stage, seed0, games):
        if self.baseline_config is None:
            raise RunnerError("stadium baseline has not been configured")
        cpus = self.free.get()
        started = time.monotonic()
        try:
            with self._match_directory() as root:
                with open(config_path) as stream:
                    candidate = self._bot_copy(root / "candidate", json.load(stream), cpus[:self.threads])
                baseline = self._bot_copy(root / "baseline", stage.config(self.baseline_config), cpus[self.threads:])
                results = []
                # Each launch changes sides; the upstream CLI has no seed interface.
                for i in range(games):
                    candidate_side = (seed0 + i) % 2 + 1
                    bots = [candidate, baseline] if candidate_side == 1 else [baseline, candidate]
                    database = root / "game.db"
                    settings = {"num_games": 1, "db_path": str(database), "min_nodes": 2000,
                                "pps": 1000000, "threads": self.threads, "thinks": stage.sims,
                                "garbage_delay": 1 if stage.rules == "botris" else 0}
                    (root / "stadium_cli_settings.json").write_text(json.dumps(settings))
                    (root / "match.json").write_text(json.dumps({"seed_counter": seed0 + i,
                        "candidate_side": candidate_side, "cpus": cpus, "stage": stage.spec}))
                    code, stdout, stderr = self._spawn([self.stadium] + bots, dict(os.environ),
                                                  cwd=str(root), cpus=cpus)
                    (root / "stdout.log").write_text(stdout, encoding="utf-8")
                    (root / "stderr.log").write_text(stderr, encoding="utf-8")
                    if code != 0 or not database.is_file():
                        raise RunnerError(f"stadium failed (exit {code}): {stderr[-2000:]}")
                    with closing(sqlite3.connect(database)) as db:
                        rows = db.execute("SELECT state FROM Data WHERE state != 'PLAYING' ORDER BY game_id").fetchall()
                    if len(rows) != 1 or rows[0][0] not in ("P1_WIN", "P2_WIN", "DRAW"):
                        raise RunnerError(f"stadium must report exactly one completed game; got {rows}")
                    outcome = rows[0][0]
                    score = 0.5 if outcome == "DRAW" else float(outcome == f"P{candidate_side}_WIN")
                    results.append({"rules": stage.rules, "match_score": score, "died": score == 0,
                                    "candidate_side": candidate_side, "outcome": outcome})
                    database.unlink()
                with self.lock:
                    self.games_played += games
                    self.chunk_times.append((time.monotonic() - started, games))
                return results
        except Exception:
            with self.lock:
                self.failures += 1
            raise
        finally:
            self.free.put(cpus)
