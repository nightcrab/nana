"""The tuning run: a ramp of stages (search budget and opponent pressure rising), PSO within each,
warm-started from the previous stage's best, release gates, checkpoints and health checks.

A run lives in tuner/runs/NAME/:
  run.json        settings: nana_play, cpus, the space, the schedule, PSO settings (fixed at start;
                  add-params extends the space)
  state.json      checkpoint, rewritten atomically after every iteration
  evals.jsonl     every iteration's evaluations (the full record)
  gates.jsonl     every release gate
  release.json    the latest config that passed a gate: a NANA_CONFIG file, ready to use
  best.json       the optimiser's current best (not gated)
  status.txt      one screen of where the run is, rewritten every iteration
  log.txt         everything printed
"""

import json
import math
import os
import shutil
import statistics
import sys
import time

import numpy as np

from .gate import run_gate
from .health import HealthError, Monitor, check_canary, check_determinism, check_plumbing
from .pso import PSO
from .runner import Runner, Stage
from .space import Space

TUNING_SEEDS = 1_000_000
GATE_SEEDS = 500_000_000
HOLDOUT_SEEDS = 900_000_000
STATE_VERSION = 1


def _write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def _read_json(path):
    with open(path) as f:
        return json.load(f)


class Run:
    def __init__(self, directory):
        self.dir = directory
        self.settings = _read_json(self.path("run.json"))
        self.space = Space(self.settings["space"])
        self.schedule = self.settings["schedule"]
        self.stages = [Stage(s) for s in self.schedule["stages"]]
        self.runner = Runner(self.settings["nana"], self.settings["cpus"], self.dir, chunk=self.settings.get("chunk", 4))
        self.state = _read_json(self.path("state.json")) if os.path.exists(self.path("state.json")) else None
        self.pso = PSO.from_state(self.state["pso"]) if self.state else None
        self.monitor = Monitor()
        self.warnings = []

    # ---- files and logging

    def path(self, name):
        return os.path.join(self.dir, name)

    def log(self, message):
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + message
        print(line, flush=True)
        with open(self.path("log.txt"), "a") as f:
            f.write(line + "\n")

    def append(self, name, record):
        with open(self.path(name), "a") as f:
            f.write(json.dumps(record) + "\n")

    def save(self):
        self.state["pso"] = self.pso.state()
        _write_json(self.path("state.json"), self.state)

    # ---- configs

    def config(self, unit):
        return self.space.config(self.space.values(unit), self.state["base_config"])

    # ---- timeline: the schedule's hours times the run's time scale (--time-scale / --total-hours)

    def time_scale(self):
        return float(self.settings.get("time_scale", 1.0))

    def stage_hours(self, stage):
        return float(stage.spec["hours"]) * self.time_scale()

    def gate_every(self):
        return float(self.schedule.get("gate_every_hours", 12)) * self.time_scale()

    def set_time_scale(self, scale=None, total_hours=None):
        """Stretch or squeeze the timeline: every stage's hours and the gate interval scale together,
        so the ramp keeps its shape. total_hours sets the scale so the stages add up to it."""
        if total_hours is not None:
            scale = total_hours / sum(float(st["hours"]) for st in self.schedule["stages"])
        if scale is None or not scale > 0:
            raise SystemExit("the time scale must be positive")
        self.settings["time_scale"] = scale
        _write_json(self.path("run.json"), self.settings)
        self.log(f"timeline x{scale:.3g}: {self.total_hours():.2f} h of stages in all "
                 f"({', '.join(f'{st.name} {self.stage_hours(st):.2f} h' for st in self.stages)}); "
                 f"gates every {self.gate_every():.2f} h")

    def total_hours(self):
        return sum(self.stage_hours(st) for st in self.stages)

    def gate_stage(self):
        gate = dict(self.schedule["gate"])
        current = self.stages[min(self.state["stage"], len(self.stages) - 1)]
        # once the ramp passes the gate's budget, gate at the stage's own settings
        if current.sims > gate["sims"]:
            gate.update({k: current.spec[k] for k in ("sims", "garbage_scale", "horizon", "env") if k in current.spec})
        gate["name"] = "gate"
        return Stage(gate)

    # ---- set up

    @staticmethod
    def create(directory, nana, cpus, space_paths, schedule_path, particles, chunk, seed):
        if os.path.exists(os.path.join(directory, "run.json")):
            raise SystemExit(f"{directory} already holds a run; use resume, or another name")
        os.makedirs(directory, exist_ok=True)
        specs = []
        for path in space_paths:
            specs += _read_json(path)
        space = Space(specs)  # validates
        settings = {"nana": os.path.abspath(nana), "cpus": cpus, "chunk": chunk, "space": space.specs(),
                    "schedule": _read_json(schedule_path), "particles": particles, "seed": seed,
                    "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        _write_json(os.path.join(directory, "run.json"), settings)
        run = Run(directory)
        base = run.runner.print_config({})
        run.state = {"version": STATE_VERSION, "stage": 0, "stage_hours": 0.0, "hours": 0.0, "iterations": 0,
                     "seed_next": TUNING_SEEDS, "gate_seed_next": GATE_SEEDS, "base_config": base,
                     "release": {"values": space.values(space.default_unit()), "source": "defaults", "gates": 0},
                     "baselines": {}, "hours_since_gate": 0.0, "stage_bests": []}
        run.pso = PSO(len(space), particles, rng_seed=seed, **settings["schedule"].get("pso", {}))
        run.pso.scatter(space.default_unit(), run.stages[0].spec.get("spread", 0.15))
        run.write_release()
        run.save()
        run.log(f"created run {directory}: {len(space)} parameters, {particles} particles, "
                f"{len(run.stages)} stages, cpus {cpus}")
        return run

    def write_release(self):
        release = self.space.config(self.state["release"]["values"], self.state["base_config"])
        _write_json(self.path("release.json"), release)

    # ---- health

    def health_checks(self):
        stage = self.stages[min(self.state["stage"], len(self.stages) - 1)]
        base = self.state["base_config"]
        center = self.config(self.pso.best()[1] if self.pso.best() else self.pso.x[0])
        random_point = self.config(np.random.default_rng(1).random(len(self.space)))
        self.log("health: plumbing (nana loads the configs we write)")
        check_plumbing(self.runner, [base, center, random_point])
        n = len(self.runner.cpus)
        self.log("health: determinism (the same seeds replay the same games)")
        check_determinism(self.runner, stage, base, HOLDOUT_SEEDS - 1000, n)
        self.log("health: canary (a broken eval must score worse)")
        b, c = check_canary(self.runner, self.stages[0], base, HOLDOUT_SEEDS - 2000, 4 * n)
        self.log(f"health: ok (baseline {b:.4f}, canary {c:.4f})")

    def baseline(self, stage):
        """The default config on the holdout seeds of this stage (computed once per stage)."""
        if stage.name not in self.state["baselines"]:
            games = self.schedule.get("holdout", {}).get("games", 4 * len(self.runner.cpus))
            scores, raw = self.runner.scores([("base", self.state["base_config"], stage, HOLDOUT_SEEDS, games)])
            self.state["baselines"][stage.name] = {"mean": statistics.mean(scores["base"]),
                                                   "died": statistics.mean(g["died"] for g in raw["base"]),
                                                   "games": games}
            self.log(f"baseline at {stage.name}: {self.state['baselines'][stage.name]['mean']:.4f} "
                     f"(dies in {self.state['baselines'][stage.name]['died']:.0%})")
        return self.state["baselines"][stage.name]

    # ---- the loop

    def iterate(self, stage):
        n = int(stage.spec.get("games_per_eval", 16))
        seed0 = self.state["seed_next"]
        self.state["seed_next"] += n
        points = self.pso.ask()
        requests = [((kind, i), self.config(unit), stage, seed0, n) for kind, i, unit in points]
        started = time.time()
        scores, games = self.runner.scores(requests)
        means = {key: statistics.mean(s) for key, s in scores.items()}
        best_before = self.pso.best()
        self.pso.tell(means)
        best = self.pso.best()

        # monitors
        scores_x = [means[("x", i)] for i in range(self.pso.n)]
        game_sd = statistics.mean(statistics.stdev(s) for s in scores.values() if len(s) > 1)
        latest = means.get(("p", best_before[0])) if best_before else None
        best_games = games.get(("p", best[0])) or games.get(("x", best[0]))
        self.warnings = self.monitor.iteration(self.pso, self.space, scores_x, game_sd, n, latest, best[2],
                                               best_games, self.state["baselines"].get(stage.name))
        if not all(math.isfinite(m) for m in means.values()):
            raise HealthError("a fitness came back NaN or infinite")

        elapsed = (time.time() - started) / 3600
        self.state["stage_hours"] += elapsed
        self.state["hours"] += elapsed
        self.state["hours_since_gate"] += elapsed
        self.state["iterations"] += 1

        record = {"iteration": self.state["iterations"], "stage": stage.name, "seed0": seed0, "games": n,
                  "scores": {f"{k}{i}": round(m, 5) for (k, i), m in means.items()},
                  "best": {"index": best[0], "mean": best[2], "evaluations": best[3],
                           "values": self.space.values(best[1])},
                  "diversity": self.pso.diversity(), "warnings": self.warnings, "seconds": elapsed * 3600}
        self.append("evals.jsonl", record)
        _write_json(self.path("best.json"), self.config(best[1]))
        return best

    def holdout(self, stage, best):
        games = self.schedule.get("holdout", {}).get("games", 4 * len(self.runner.cpus))
        scores, _ = self.runner.scores([("best", self.config(best[1]), stage, HOLDOUT_SEEDS, games)])
        base = self.state["baselines"][stage.name]["mean"]
        value = statistics.mean(scores["best"])
        self.log(f"holdout at {stage.name}: best {value:.4f} vs baseline {base:.4f} ({value - base:+.4f})")
        self.state.setdefault("holdouts", []).append({"iteration": self.state["iterations"], "stage": stage.name,
                                                      "best": value, "baseline": base})
        if value < base and best[2] > base:
            self.warnings.append("holdout: the best beats the baseline on tuning seeds but not on held-out ones")

    def gate(self, best, reason):
        stage = self.gate_stage()
        settings = self.schedule["gate"]
        candidate_values = self.space.values(best[1])
        candidate = self.space.config(candidate_values, self.state["base_config"])
        release = self.space.config(self.state["release"]["values"], self.state["base_config"])
        if candidate == release:
            self.log(f"gate ({reason}): the best is the release; nothing to test")
            self.state["hours_since_gate"] = 0.0
            return
        self.log(f"gate ({reason}) at {stage.sims} sims, garbage x{stage.garbage_scale}")
        started = time.time()
        result = run_gate(self.runner, stage, candidate, release, self.state["gate_seed_next"],
                          delta=settings.get("delta", 0.02), alpha=settings.get("alpha", 0.05),
                          beta=settings.get("beta", 0.05), max_games=settings.get("max_games", 1600), log=self.log)
        self.state["gate_seed_next"] = result["seed_end"]
        elapsed = (time.time() - started) / 3600
        self.state["hours"] += elapsed
        self.state["hours_since_gate"] = 0.0
        result.update({"reason": reason, "iteration": self.state["iterations"], "hours": self.state["hours"],
                       "values": candidate_values})
        self.append("gates.jsonl", result)
        if result["verdict"] == "accept":
            self.state["release"] = {"values": candidate_values, "source": f"{reason}, iteration {self.state['iterations']}",
                                     "gates": self.state["release"]["gates"] + 1, "fitness": result["candidate"]}
            self.write_release()
            self.log(f"gate: released ({result['candidate']:.4f} vs {result['release']:.4f} in {result['games']} games)")
        else:
            self.log(f"gate: {result['verdict']}, the release stays ({result['candidate']:.4f} vs "
                     f"{result['release']:.4f} in {result['games']} games)")

    def next_stage(self, best):
        stage = self.stages[self.state["stage"]]
        self.state["stage_bests"].append({"stage": stage.name, "values": self.space.values(best[1]),
                                          "mean": best[2], "iteration": self.state["iterations"]})
        self.gate(best, f"end of {stage.name}")
        self.state["stage"] += 1
        self.state["stage_hours"] = 0.0
        if self.state["stage"] < len(self.stages):
            following = self.stages[self.state["stage"]]
            self.pso.scatter(best[1], following.spec.get("spread", 0.1))
            self.monitor = Monitor()
            self.log(f"stage {following.name}: {following.sims} sims, garbage x{following.garbage_scale}, "
                     f"{self.stage_hours(following):.2f} h, warm-started from {stage.name}'s best")
        self.save()

    def status(self, best):
        stage = self.stages[min(self.state["stage"], len(self.stages) - 1)]
        base = self.state["baselines"].get(stage.name, {}).get("mean")
        lines = [f"run {self.dir}  ({self.state['hours']:.2f} h, {self.state['iterations']} iterations, "
                 f"{self.runner.games_played} games this session, {self.runner.failures} game failures)",
                 f"stage {self.state['stage'] + 1}/{len(self.stages)} {stage.name}: {stage.sims} sims, garbage "
                 f"x{stage.garbage_scale}, {self.state['stage_hours']:.2f}/{self.stage_hours(stage):.2f} h"
                 + (f" (timeline x{self.time_scale():.3g})" if self.time_scale() != 1 else "")]
        if best:
            lines.append(f"best: {best[2]:.4f} over {best[3]} evaluations"
                         + (f" (baseline {base:.4f}, {best[2] - base:+.4f})" if base is not None else ""))
        lines.append(f"release: {self.state['release']['source']} ({self.state['release']['gates']} gates passed)"
                     + f"; next gate in {max(0, self.gate_every() - self.state['hours_since_gate']):.1f} h")
        lines.append(f"diversity {self.pso.diversity():.3f}")
        lines += ["WARNING " + w for w in self.warnings] or ["no warnings"]
        with open(self.path("status.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

    def run(self, max_hours=None):
        self.health_checks()
        started = self.state["hours"]
        holdout_every = self.schedule.get("holdout", {}).get("every", 20)
        while self.state["stage"] < len(self.stages):
            stage = self.stages[self.state["stage"]]
            self.baseline(stage)
            if self.state["stage_hours"] >= self.stage_hours(stage):
                self.next_stage(self.pso.best())
                continue
            best = self.iterate(stage)
            if self.state["iterations"] % holdout_every == 0:
                self.holdout(stage, best)
            for w in self.warnings:
                self.log("WARNING " + w)
            if self.state["hours_since_gate"] >= self.gate_every():
                self.gate(best, f"periodic, {stage.name}")
            self.save()
            self.status(best)
            if self.state["iterations"] % 10 == 0:
                self.log(f"{stage.name} iteration {self.state['iterations']}: best {best[2]:.4f} "
                         f"({best[3]} evals), diversity {self.pso.diversity():.3f}, {self.state['hours']:.2f} h")
            if max_hours is not None and self.state["hours"] - started >= max_hours:
                self.log(f"stopping after {max_hours} h as asked (resume to continue)")
                return
        self.log("the ramp is done; release.json is the result")

    # ---- restarts

    def restart_stage(self, k, source):
        if not 0 <= k < len(self.stages):
            raise SystemExit(f"no stage {k}; stages are 0..{len(self.stages) - 1}")
        if source == "release":
            values = self.state["release"]["values"]
        elif source == "best":
            values = self.space.values(self.pso.best()[1])
        else:
            loaded = _read_json(source)
            values = {p.name: (loaded[p.key][p.index] if p.index is not None else loaded[p.key])
                      for p in self.space.params if p.key in loaded}
        self.state["stage"] = k
        self.state["stage_hours"] = 0.0
        self.pso.scatter(self.space.unit(values), self.stages[k].spec.get("spread", 0.1))
        self.monitor = Monitor()
        self.save()
        self.log(f"restarting the ramp at stage {k} ({self.stages[k].name}) from {source}")

    def add_params(self, space_path):
        extra = Space(_read_json(space_path))
        grown = self.space.extend(extra)
        if len(grown) == len(self.space):
            raise SystemExit("those parameters are all in the space already")
        added = len(grown) - len(self.space)
        self.settings["space"] = grown.specs()
        _write_json(self.path("run.json"), self.settings)
        self.space = grown
        self.pso.grow(len(grown), grown.default_unit())
        # the release and stage bests keep their values; the new parameters stay at defaults there
        self.save()
        self.log(f"added {added} parameters: now {len(grown)}")
