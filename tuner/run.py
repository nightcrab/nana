"""A tuning run: PSO over nana's constants at one search budget, picked to fit a time budget.

start runs the health checks, then calibrates: it times games at a ladder of sims counts, with
configs scattered the way the swarm starts, and picks the largest sims count whose whole run fits
--budget-hours (or uses --sims as given). Then a fixed number of PSO iterations at that budget,
a holdout against the defaults every few iterations, and one release gate at the end.

Exploration comes first and exploitation last: the swarm starts widely scattered around the
defaults (a weak prior), and the inertia falls linearly over the iterations, so the swarm settles.

A run lives in tuner/runs/NAME/:
  run.json        settings, fixed at start
  state.json      checkpoint, rewritten atomically after every iteration
  plan.json       the calibration: seconds per game at each sims count tried, and the plan
  evals.jsonl     every iteration's evaluations (the full record)
  gates.jsonl     the release gate
  release.json    the config that passed the gate (the defaults until one does): a NANA_CONFIG file
  best.json       the optimiser's current best (not gated)
  status.txt      one screen of where the run is, rewritten every iteration
  log.txt         everything printed
"""

import json
import math
import os
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
CALIBRATION_SEEDS = 800_000_000
HOLDOUT_SEEDS = 900_000_000
STATE_VERSION = 2

# the sims counts calibration tries, cheapest first
SIMS_LADDER = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
# the opponent's garbage, eased at small budgets so a weak search survives long enough to be scored:
# (sims, garbage_scale), interpolated in log2(sims), and 1.0 from 8192 sims up
GARBAGE = [(32, 0.25), (128, 0.4), (512, 0.6), (2048, 0.8), (8192, 1.0)]
# health checks play at a small budget: they test the plumbing, not the search
CHECK_STAGE = {"name": "check", "sims": 32, "garbage_scale": 0.25}

DEFAULTS = {
    "iterations": 15,
    "particles": 24,
    "games_per_eval": 16,
    "spread": 0.3,
    "horizon": 100,
    "env": "hmm",
    "pso": {"inertia_start": 0.9, "inertia_end": 0.4, "cognitive": 1.5, "social": 1.5, "vmax": 0.2},
    "gate": {"delta": 0.02, "alpha": 0.05, "beta": 0.05, "max_games": 400},
    "holdout": {"games": 32, "every": 5},
}


def garbage_for(sims):
    if sims <= GARBAGE[0][0]:
        return GARBAGE[0][1]
    for (s0, g0), (s1, g1) in zip(GARBAGE, GARBAGE[1:]):
        if sims <= s1:
            t = (math.log2(sims) - math.log2(s0)) / (math.log2(s1) - math.log2(s0))
            return round(g0 + t * (g1 - g0), 4)
    return GARBAGE[-1][1]


def _clock(seconds):
    """m:ss, or h:mm:ss from an hour"""
    seconds = int(seconds)
    h, m, s = seconds // 3600, seconds // 60 % 60, seconds % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def _read_json(path):
    with open(path) as f:
        return json.load(f)


class Run:
    def __init__(self, directory, verbose=False):
        self.dir = directory
        self.verbose = verbose
        self.settings = _read_json(self.path("run.json"))
        self.space = Space(self.settings["space"])
        self.runner = Runner(self.settings["nana"], self.settings["cpus"], self.dir, chunk=self.settings.get("chunk", 4))
        self.state = _read_json(self.path("state.json")) if os.path.exists(self.path("state.json")) else None
        if self.state and self.state.get("version") != STATE_VERSION:
            raise SystemExit(f"{directory} was made by an older tuner (state version {self.state.get('version')}); "
                             "start a new run")
        self.pso = PSO.from_state(self.state["pso"]) if self.state else None
        self.monitor = Monitor()
        self.warnings = []
        self.warned = set()
        # the live line: what the run is doing now, redrawn in place (only on a terminal)
        self.live = sys.stdout.isatty()
        self.doing = ""
        self.doing_since = time.monotonic()
        self.live_width = 0
        self.runner.progress = self.progress

    # ---- files and logging

    def path(self, name):
        return os.path.join(self.dir, name)

    def log(self, message, show=False):
        """everything goes to log.txt; the console gets only what show marks (progress and trouble),
        or everything with --verbose"""
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + message
        if show or self.verbose:
            self.clear_live()
            print(line, flush=True)
        with open(self.path("log.txt"), "a") as f:
            f.write(line + "\n")

    def activity(self, label):
        """name what the run is doing now, for the live line"""
        self.doing = label
        self.doing_since = time.monotonic()

    def progress(self, done, total):
        if not self.live or not self.doing:
            return
        elapsed = time.monotonic() - self.doing_since
        text = f"  {self.doing}: {done}/{total} games, {_clock(elapsed)} elapsed"
        sys.stdout.write("\r" + text.ljust(self.live_width))
        sys.stdout.flush()
        self.live_width = len(text)

    def clear_live(self):
        if self.live_width:
            sys.stdout.write("\r" + " " * self.live_width + "\r")
            sys.stdout.flush()
            self.live_width = 0

    def append(self, name, record):
        with open(self.path(name), "a") as f:
            f.write(json.dumps(record) + "\n")

    def save(self):
        self.state["pso"] = self.pso.state()
        _write_json(self.path("state.json"), self.state)

    def account(self, started):
        """add the wall time since started (time.monotonic) to the run's hours"""
        self.state["hours"] += (time.monotonic() - started) / 3600

    # ---- configs and the game settings

    def config(self, unit):
        return self.space.config(self.space.values(unit), self.state["base_config"])

    def make_stage(self, sims):
        return Stage({"name": f"s{sims}", "sims": sims, "garbage_scale": garbage_for(sims),
                      "horizon": self.settings["horizon"], "env": self.settings["env"]})

    def stage(self):
        return self.make_stage(self.state["plan"]["sims"])

    # ---- set up

    @staticmethod
    def create(directory, nana, cpus, space_paths, options, verbose=False):
        if os.path.exists(os.path.join(directory, "run.json")):
            raise SystemExit(f"{directory} already holds a run; use resume, or another name")
        if not os.path.isfile(nana):
            raise SystemExit(f"no nana_emulate at {nana}")
        os.makedirs(directory, exist_ok=True)
        specs = []
        for path in space_paths:
            specs += _read_json(path)
        space = Space(specs)  # validates
        settings = json.loads(json.dumps(DEFAULTS))
        for key, value in options.items():
            if value is None:
                continue
            if key in ("gate_games", "holdout_every"):
                settings["gate" if key == "gate_games" else "holdout"][
                    "max_games" if key == "gate_games" else "every"] = value
            else:
                settings[key] = value
        if not settings.get("budget_hours") and not settings.get("sims"):
            raise SystemExit("give --budget-hours (the tuner picks the sims) or --sims")
        settings.update({"nana": os.path.abspath(nana), "cpus": cpus, "space": space.specs(),
                         "created": time.strftime("%Y-%m-%d %H:%M:%S")})
        _write_json(os.path.join(directory, "run.json"), settings)
        run = Run(directory, verbose)
        base = run.runner.print_config({})
        run.state = {"version": STATE_VERSION, "phase": "calibrate", "iterations": 0, "hours": 0.0,
                     "seed_next": TUNING_SEEDS, "gate_seed_next": GATE_SEEDS, "base_config": base,
                     "release": {"values": space.values(space.default_unit()), "source": "defaults", "gates": 0},
                     "plan": None, "baseline": None, "holdouts": [], "iteration_seconds": []}
        pso = settings["pso"]
        run.pso = PSO(len(space), settings["particles"], rng_seed=settings.get("seed", 1),
                      inertia=pso["inertia_start"], cognitive=pso["cognitive"], social=pso["social"], vmax=pso["vmax"])
        run.pso.scatter(space.default_unit(), settings["spread"])
        run.write_release()
        run.save()
        run.log(f"created run {directory}: {len(space)} parameters, {settings['particles']} particles, "
                f"{settings['iterations']} iterations, cpus {cpus}")
        return run

    def write_release(self):
        release = self.space.config(self.state["release"]["values"], self.state["base_config"])
        _write_json(self.path("release.json"), release)

    # ---- health

    def health_checks(self):
        self.activity("health checks")
        started = time.monotonic()
        stage = Stage(dict(CHECK_STAGE, horizon=self.settings["horizon"], env=self.settings["env"]))
        base = self.state["base_config"]
        center = self.config(self.pso.best()[1] if self.pso.best() else self.pso.x[0])
        random_point = self.config(np.random.default_rng(1).random(len(self.space)))
        self.log("health: plumbing (nana loads the configs we write)")
        check_plumbing(self.runner, [base, center, random_point])
        n = len(self.runner.cpus)
        self.log("health: determinism (the same seeds replay the same games)")
        check_determinism(self.runner, stage, base, HOLDOUT_SEEDS - 1000, n)
        self.log("health: canary (a broken eval must score worse)")
        b, c = check_canary(self.runner, stage, base, HOLDOUT_SEEDS - 2000, 4 * n)
        self.log(f"health: ok (baseline {b:.4f}, canary {c:.4f})")
        if not self.verbose:
            self.log("health checks ok", show=True)
        self.account(started)

    # ---- calibration and the plan

    def time_games(self, sims):
        """Core-seconds per game at this many sims: two waves of games over the cpus, with the
        swarm's starting configs, so the games are as long as the run's will be at the start."""
        self.activity(f"timing games at {sims} sims")
        stage = self.make_stage(sims)
        k = 2 * len(self.runner.cpus)
        requests = [(("cal", i), self.config(self.pso.x[i % self.pso.n]), stage, CALIBRATION_SEEDS + i, 1)
                    for i in range(k)]
        started = time.monotonic()
        self.runner.play(requests)
        wall = time.monotonic() - started
        return wall * len(self.runner.cpus) / k

    def projection(self, core_seconds):
        """Hours for the whole run at this cost per game: the iterations (the first scores only the
        particles, later ones their bests too), the baseline and holdouts, and the gate at its limit."""
        s = self.settings
        per_game = core_seconds / len(self.runner.cpus)
        p, g, n = s["particles"], s["games_per_eval"], s["iterations"]
        iteration = 2 * p * g * per_game
        tune = p * g * per_game + (n - 1) * iteration
        every = s["holdout"]["every"]
        holdouts = n // every + (1 if n % every else 0)
        checks = (holdouts + 1) * s["holdout"]["games"] * per_game
        gate = 2 * s["gate"]["max_games"] * per_game
        hours = {"iteration": iteration, "tune": tune, "checks": checks, "gate": gate}
        hours = {k: v / 3600 for k, v in hours.items()}
        hours["total"] = hours["tune"] + hours["checks"] + hours["gate"]
        return hours

    def calibrate(self):
        s = self.settings
        forced = s.get("sims")
        budget = s.get("budget_hours")
        ladder = [forced] if forced else SIMS_LADDER
        measured = {}
        chosen = None
        for sims in ladder:
            started = time.monotonic()
            if chosen and budget:
                # a rung costs about twice the last: don't spend time measuring one that clearly won't fit
                if self.projection(2 * measured[chosen])["total"] > 1.3 * (budget - self.state["hours"]):
                    break
            core_seconds = self.time_games(sims)
            self.account(started)
            measured[sims] = core_seconds
            p = self.projection(core_seconds)
            self.log(f"calibrate: {sims} sims, {core_seconds:.1f} core-s per game: {p['total']:.2f} h "
                     f"({p['iteration'] * 60:.1f} min per iteration)")
            if forced or p["total"] <= budget - self.state["hours"]:
                chosen = sims
            else:
                break
        if chosen is None:
            p = self.projection(measured[ladder[0]])
            raise SystemExit(f"even {ladder[0]} sims needs about {p['total']:.1f} h after calibration, more than "
                             f"the {budget} h budget leaves; raise --budget-hours, or lower --iterations, "
                             "--particles or --games")
        p = self.projection(measured[chosen])
        if forced and budget and p["total"] > budget - self.state["hours"]:
            self.log(f"WARNING {forced} sims needs about {p['total']:.2f} h, more than the {budget} h budget leaves", show=True)
        self.state["plan"] = {"sims": chosen, "garbage_scale": garbage_for(chosen),
                              "core_seconds_per_game": measured[chosen], "projected_hours": p,
                              "calibration_hours": self.state["hours"]}
        _write_json(self.path("plan.json"), {"measured": {str(k): v for k, v in measured.items()},
                                             "plan": self.state["plan"], "budget_hours": budget})
        self.log(f"plan: {chosen} sims, garbage x{garbage_for(chosen)}, {s['iterations']} iterations of "
                 f"~{p['iteration'] * 60:.1f} min, about {p['total']:.2f} h in all (tuning {p['tune']:.2f}, "
                 f"holdouts {p['checks']:.2f}, gate at most {p['gate']:.2f})"
                 + (f"; the budget is {budget} h" if budget else ""))
        if not self.verbose:
            self.log(f"plan: {chosen} sims, {s['iterations']} iterations, about {p['total']:.1f} h"
                     + (f" of the {budget:g} h budget" if budget else "") + f"; {self.eta(p['total'])}", show=True)
        self.state["phase"] = "tune"
        self.save()

    # ---- the loop

    def baseline(self):
        """The defaults on the holdout seeds (once)."""
        if self.state["baseline"] is None:
            started = time.monotonic()
            games = self.settings["holdout"]["games"]
            self.activity("scoring the defaults")
            scores, raw = self.runner.scores([("base", self.state["base_config"], self.stage(), HOLDOUT_SEEDS, games)])
            self.state["baseline"] = {"mean": statistics.mean(scores["base"]),
                                      "died": statistics.mean(g["died"] for g in raw["base"]), "games": games}
            self.account(started)
            self.log(f"baseline: {self.state['baseline']['mean']:.4f} (dies in {self.state['baseline']['died']:.0%})")
            self.save()
        return self.state["baseline"]

    def iterate(self):
        stage = self.stage()
        s = self.settings
        n = s["games_per_eval"]
        # inertia falls linearly over the run: the swarm explores first and settles at the end
        k, total = self.state["iterations"], s["iterations"]
        frac = k / (total - 1) if total > 1 else 1.0
        self.pso.inertia = s["pso"]["inertia_start"] + frac * (s["pso"]["inertia_end"] - s["pso"]["inertia_start"])
        seed0 = self.state["seed_next"]
        self.state["seed_next"] += n
        points = self.pso.ask()
        eta = time.strftime("%H:%M", time.localtime(time.time() + self.hours_left() * 3600))
        self.activity(f"iteration {k + 1}/{total} (run done by {eta})")
        requests = [((kind, i), self.config(unit), stage, seed0, n) for kind, i, unit in points]
        started = time.monotonic()
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
                                               best_games, self.state["baseline"])
        if not all(math.isfinite(m) for m in means.values()):
            raise HealthError("a fitness came back NaN or infinite")

        seconds = time.monotonic() - started
        self.state["hours"] += seconds / 3600
        self.state["iterations"] += 1
        self.state["iteration_seconds"].append(seconds)

        record = {"iteration": self.state["iterations"], "sims": stage.sims, "seed0": seed0, "games": n,
                  "inertia": self.pso.inertia,
                  "scores": {f"{k}{i}": round(m, 5) for (k, i), m in means.items()},
                  "best": {"index": best[0], "mean": best[2], "evaluations": best[3],
                           "values": self.space.values(best[1])},
                  "diversity": self.pso.diversity(), "warnings": self.warnings, "seconds": seconds}
        self.append("evals.jsonl", record)
        _write_json(self.path("best.json"), self.config(best[1]))
        return best

    def holdout(self, best):
        started = time.monotonic()
        games = self.settings["holdout"]["games"]
        self.activity("holdout")
        scores, _ = self.runner.scores([("best", self.config(best[1]), self.stage(), HOLDOUT_SEEDS, games)])
        base = self.state["baseline"]["mean"]
        value = statistics.mean(scores["best"])
        self.account(started)
        self.log(f"holdout: best {value:.4f} vs baseline {base:.4f} ({value - base:+.4f})")
        self.state["holdouts"].append({"iteration": self.state["iterations"], "best": value, "baseline": base})
        if value < base and best[2] > base:
            self.warnings.append("holdout: the best beats the baseline on tuning seeds but not on held-out ones")

    def gate(self, best):
        stage = self.stage()
        settings = self.settings["gate"]
        candidate_values = self.space.values(best[1])
        candidate = self.space.config(candidate_values, self.state["base_config"])
        release = self.space.config(self.state["release"]["values"], self.state["base_config"])
        if candidate == release:
            self.log("result: the best is the release; nothing to test", show=True)
            return
        self.log(f"gate at {stage.sims} sims, garbage x{stage.garbage_scale}")
        self.activity("release gate")
        started = time.monotonic()
        result = run_gate(self.runner, stage, candidate, release, self.state["gate_seed_next"],
                          delta=settings["delta"], alpha=settings["alpha"], beta=settings["beta"],
                          max_games=settings["max_games"], log=self.log)
        self.state["gate_seed_next"] = result["seed_end"]
        self.account(started)
        result.update({"iteration": self.state["iterations"], "hours": self.state["hours"], "values": candidate_values})
        self.append("gates.jsonl", result)
        if result["verdict"] == "accept":
            self.state["release"] = {"values": candidate_values, "source": f"gate after iteration {self.state['iterations']}",
                                     "gates": self.state["release"]["gates"] + 1, "fitness": result["candidate"]}
            self.write_release()
            self.log(f"result: released ({result['candidate']:.4f} vs {result['release']:.4f} in {result['games']} games)",
                     show=True)
        else:
            self.log(f"result: gate {result['verdict']}, the release stays ({result['candidate']:.4f} vs "
                     f"{result['release']:.4f} in {result['games']} games)", show=True)

    def hours_left(self):
        """the iterations left at the recent pace, plus the gate at its limit"""
        plan = self.state["plan"]
        if not plan or self.state["phase"] == "done":
            return 0.0
        recent = self.state["iteration_seconds"][-3:]
        per = statistics.mean(recent) / 3600 if recent else plan["projected_hours"]["iteration"]
        return (self.settings["iterations"] - self.state["iterations"]) * per + plan["projected_hours"]["gate"]

    def eta(self, hours):
        finish = time.strftime("%a %H:%M", time.localtime(time.time() + hours * 3600))
        return f"done by {finish} at the latest ({hours:.1f} h left)"

    def status(self, best):
        s, plan = self.settings, self.state["plan"]
        done, total = self.state["iterations"], s["iterations"]
        lines = [f"run {self.dir}  ({self.state['hours']:.2f} h, {self.runner.games_played} games this session, "
                 f"{self.runner.failures} game failures)"]
        if plan:
            lines.append(f"{self.state['phase']}: {plan['sims']} sims, garbage x{plan['garbage_scale']}, iteration "
                         f"{done}/{total}" + ("" if self.state["phase"] == "done" else
                                             f", {self.eta(self.hours_left())}"))
        if best:
            base = (self.state["baseline"] or {}).get("mean")
            lines.append(f"best: {best[2]:.4f} over {best[3]} evaluations"
                         + (f" (baseline {base:.4f}, {best[2] - base:+.4f})" if base is not None else ""))
        if self.state["holdouts"]:
            h = self.state["holdouts"][-1]
            lines.append(f"holdout after iteration {h['iteration']}: {h['best']:.4f} vs baseline {h['baseline']:.4f} "
                         f"({h['best'] - h['baseline']:+.4f})")
        lines.append(f"release: {self.state['release']['source']} ({self.state['release']['gates']} gates passed)")
        lines.append(f"diversity {self.pso.diversity():.3f}, inertia {self.pso.inertia:.2f}")
        lines += ["WARNING " + w for w in self.warnings] or ["no warnings"]
        with open(self.path("status.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

    def run(self, plan_only=False):
        if self.state["phase"] == "done":
            self.log("the run is done; release.json is the result", show=True)
            return
        self.health_checks()
        if self.state["phase"] == "calibrate":
            self.calibrate()
            self.status(self.pso.best())
            if plan_only:
                self.log("stopping after the plan as asked (resume to tune)", show=True)
                return
        total = self.settings["iterations"]
        every = self.settings["holdout"]["every"]
        if self.state["phase"] == "tune":
            self.baseline()
            while self.state["iterations"] < total:
                best = self.iterate()
                self.log(f"iteration {self.state['iterations']}/{total}: best {best[2]:.4f} ({best[3]} evals), "
                         f"diversity {self.pso.diversity():.3f}, {self.state['iteration_seconds'][-1] / 60:.1f} min")
                if self.state["iterations"] % every == 0 or self.state["iterations"] == total:
                    self.holdout(best)
                if not self.verbose:
                    self.log(f"iteration {self.state['iterations']}/{total}, {self.eta(self.hours_left())}", show=True)
                for w in self.warnings:
                    # on the console, each kind of warning once; log.txt keeps every one
                    kind = w.split(":")[0]
                    self.log("WARNING " + w, show=kind not in self.warned)
                    self.warned.add(kind)
                self.save()
                self.status(best)
            self.state["phase"] = "gate"
            self.save()
        if self.state["phase"] == "gate":
            self.gate(self.pso.best())
            self.state["phase"] = "done"
            self.save()
            self.status(self.pso.best())
        self.log(f"done in {self.state['hours']:.2f} h; release.json is the result", show=True)
