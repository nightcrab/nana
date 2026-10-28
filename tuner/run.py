import json
import math
import os
import statistics
import sys
import time

import numpy as np

from .gate import run_gate
from .health import HealthError, Monitor, check_canary, check_determinism, check_plumbing
from .optimizer import create_optimizer, optimizer_defaults, restore_optimizer
from .runner import Runner, Stage
from .space import Space

TUNING_SEEDS = 1_000_000
GATE_SEEDS = 500_000_000
CALIBRATION_SEEDS = 800_000_000
HOLDOUT_SEEDS = 900_000_000
STATE_VERSION = 2

SIMS_LADDER = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
GARBAGE = [(32, 0.25), (128, 0.4), (512, 0.6), (2048, 0.8), (8192, 1.0)]
CHECK_STAGE = {"name": "check", "sims": 32, "garbage_scale": 0.25}

DEFAULTS = {
    "optimizer": "pso",
    "threads": 1,
    "objective": "strength",
    "iterations": 15,
    "games_per_eval": 32,
    "horizon": 100,
    "rules": "tetrio",
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
        self.settings.setdefault("objective", "strength")
        self.settings.setdefault("threads", 1)
        self.settings.setdefault("optimizer", "pso")
        self.space = Space(self.settings["space"])
        self.runner = Runner(self.settings["nana"], self.settings["cpus"], self.dir,
                             chunk=self.settings.get("chunk", 4), threads=self.settings["threads"])
        self.state = _read_json(self.path("state.json")) if os.path.exists(self.path("state.json")) else None
        if self.state and self.state.get("version") != STATE_VERSION:
            raise SystemExit(f"{directory} was made by an older tuner (state version {self.state.get('version')}); "
                             "start a new run")
        self.optimizer = restore_optimizer(self.settings, self.state) if self.state else None
        self.monitor = Monitor()
        self.warnings = []
        self.warned = set()
        self.live = sys.stdout.isatty()
        self.doing = ""
        self.doing_since = time.monotonic()
        self.live_width = 0
        self.runner.progress = self.progress

    def path(self, name):
        return os.path.join(self.dir, name)

    def log(self, message, show=False):
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + message
        if show or self.verbose:
            self.clear_live()
            print(line, flush=True)
        with open(self.path("log.txt"), "a") as f:
            f.write(line + "\n")

    def activity(self, label):
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
        self.state["optimizer"] = self.optimizer.state()
        self.state.pop("pso", None)
        _write_json(self.path("state.json"), self.state)

    def account(self, started):
        self.state["hours"] += (time.monotonic() - started) / 3600

    def config(self, unit):
        return self.space.config(self.space.values(unit), self.state["base_config"])

    def make_stage(self, sims):
        return Stage({"name": f"s{sims}", "sims": sims, "garbage_scale": garbage_for(sims),
                      "objective": self.settings["objective"],
                      "horizon": self.settings["horizon"], "env": self.settings.get("env", "hmm"), "rules": self.rules()})

    def rules(self):
        return self.settings.get("rules", "tetrio")

    def stage(self):
        return self.make_stage(self.state["plan"]["sims"])

    @staticmethod
    def create(directory, nana, cpus, space_paths, options, verbose=False):
        if os.path.exists(os.path.join(directory, "run.json")):
            raise SystemExit(f"{directory} already holds a run; use resume, or another name")
        if not os.path.isfile(nana):
            raise SystemExit(f"no nana_emulate at {nana}")
        try:
            Runner.validate_cpus(cpus, 1 if options.get("threads") is None else options["threads"])
        except ValueError as e:
            raise SystemExit(str(e)) from e
        os.makedirs(directory, exist_ok=True)
        specs = []
        for path in space_paths:
            specs += _read_json(path)
        space = Space(specs)
        settings = json.loads(json.dumps(DEFAULTS))
        settings.update(json.loads(json.dumps(optimizer_defaults(options.get("optimizer") or "pso"))))
        if options.get("objective") == "sims-per-second":
            settings["games_per_eval"] = 4
            settings["holdout"]["games"] = 8
            settings["gate"]["max_games"] = 32
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
        for key in ("games_per_eval", "iterations", "horizon", "chunk"):
            if settings.get(key, 4) < 1:
                raise SystemExit(f"{key} must be positive")
        try:
            optimizer = create_optimizer(settings, space.default_unit())
        except ValueError as e:
            raise SystemExit(str(e)) from e
        settings.update({"nana": os.path.abspath(nana), "cpus": cpus, "space": space.specs(),
                         "created": time.strftime("%Y-%m-%d %H:%M:%S")})
        _write_json(os.path.join(directory, "run.json"), settings)
        run = Run(directory, verbose)
        base = run.runner.print_config({"eval": settings["eval"]} if settings.get("eval") else {})
        run.state = {"version": STATE_VERSION, "phase": "calibrate", "iterations": 0, "hours": 0.0,
                     "seed_next": TUNING_SEEDS, "gate_seed_next": GATE_SEEDS, "base_config": base,
                     "release": {"values": space.values(space.default_unit()), "source": "defaults", "gates": 0},
                     "plan": None, "baseline": None, "holdouts": [], "iteration_seconds": []}
        run.optimizer = optimizer
        run.write_release()
        run.save()
        run.log(f"created run {directory}: {settings['objective']}, {settings['rules']} rules, {len(space)} parameters, {optimizer.name} optimizer, "
                f"{settings['iterations']} iterations, {settings['threads']} threads per game, cpus {cpus}")
        return run

    def write_release(self):
        release = self.space.config(self.state["release"]["values"], self.state["base_config"])
        _write_json(self.path("release.json"), release)

    def health_checks(self):
        self.activity("health checks")
        started = time.monotonic()
        stage = Stage(dict(CHECK_STAGE, horizon=self.settings["horizon"], env=self.settings.get("env", "hmm"), rules=self.rules()))
        base = self.state["base_config"]
        best = self.optimizer.best()
        center = self.config(best.unit if best else self.optimizer.sample_points()[0])
        random_point = self.config(np.random.default_rng(1).random(len(self.space)))
        self.log("health: plumbing (nana loads the configs we write)")
        check_plumbing(self.runner, [base, center, random_point])
        n = self.runner.workers
        if self.runner.threads == 1:
            self.log("health: determinism (the same seeds replay the same games)")
            check_determinism(self.runner, stage, base, HOLDOUT_SEEDS - 1000, n)
        else:
            self.log("health: exact seed replay skipped (multithreaded search depends on scheduling)")
        if self.settings["objective"] == "sims-per-second":
            # Timing is noisy; determinism above deliberately checks attack, not speed.
            self.runner.scores([("speed", base, self.make_stage(32), HOLDOUT_SEEDS - 2000, 1)])
            self.log("health checks ok (config plumbing, throughput metrics)", show=True)
            self.account(started)
            return
        self.log("health: canary (a broken eval must score worse)")
        b, c = check_canary(self.runner, stage, base, HOLDOUT_SEEDS - 2000, 4 * n)
        self.log(f"health: ok (baseline {b:.4f}, canary {c:.4f})")
        if not self.verbose:
            self.log("health checks ok", show=True)
        self.account(started)

    def time_games(self, sims):
        self.activity(f"timing games at {sims} sims")
        stage = self.make_stage(sims)
        k = max(4 if self.settings["objective"] == "sims-per-second" else 16, 2 * self.runner.workers)
        points = self.optimizer.sample_points()
        requests = [(("cal", i), self.config(points[i % len(points)]), stage, CALIBRATION_SEEDS + i, 1)
                    for i in range(k)]
        self.runner.chunk_times = []
        self.runner.play(requests)
        return sum(t for t, _ in self.runner.chunk_times) / sum(g for _, g in self.runner.chunk_times)

    def projection(self, core_seconds):
        s = self.settings
        cpus = self.runner.workers
        per_game = core_seconds / cpus
        tail = s.get("chunk", 4) * core_seconds
        g, n = s["games_per_eval"], s["iterations"]
        counts = self.optimizer.evaluation_counts(n)
        iteration = counts[-1] * g * per_game + tail
        tune = sum(count * g * per_game + tail for count in counts)
        every = s["holdout"]["every"]
        holdouts = n // every + (1 if n % every else 0)
        checks = (holdouts + 1) * (s["holdout"]["games"] * per_game + tail)
        gate_batches = math.ceil(s["gate"]["max_games"] / (4 * cpus))
        gate = 2 * s["gate"]["max_games"] * per_game + gate_batches * tail
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
        self.state["plan"] = {"objective": s["objective"], "rules": self.rules(), "sims": chosen, "garbage_scale": garbage_for(chosen),
                              "core_seconds_per_game": measured[chosen], "projected_hours": p,
                              "calibration_hours": self.state["hours"]}
        _write_json(self.path("plan.json"), {"measured": {str(k): v for k, v in measured.items()},
                                             "plan": self.state["plan"], "budget_hours": budget})
        self.log(f"plan: {self.rules()} rules, {chosen} sims, garbage x{garbage_for(chosen)}, {s['iterations']} iterations of "
                 f"~{p['iteration'] * 60:.1f} min, about {p['total']:.2f} h in all (tuning {p['tune']:.2f}, "
                 f"holdouts {p['checks']:.2f}, gate at most {p['gate']:.2f})"
                 + (f"; the budget is {budget} h" if budget else ""))
        if not self.verbose:
            self.log(f"plan: {self.rules()} rules, {chosen} sims, {s['iterations']} iterations, about {p['total']:.1f} h"
                     + (f" of the {budget:g} h budget" if budget else "") + f"; {self.eta(p['total'])}", show=True)
        self.state["phase"] = "tune"
        self.save()

    def baseline(self):
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
        k, total = self.state["iterations"], s["iterations"]
        seed0 = self.state["seed_next"]
        self.state["seed_next"] += n
        points = self.optimizer.ask()
        eta = time.strftime("%H:%M", time.localtime(time.time() + self.hours_left() * 3600))
        self.activity(f"iteration {k + 1}/{total} (run done by {eta})")
        requests = [(point.id, self.config(point.unit), stage, seed0, n) for point in points]
        started = time.monotonic()
        scores, games = self.runner.scores(requests)
        means = {key: statistics.mean(s) for key, s in scores.items()}
        if not all(math.isfinite(m) for m in means.values()):
            raise HealthError("a fitness came back NaN or infinite")
        feedback = self.optimizer.tell(scores)
        best = self.optimizer.best()
        game_sd = statistics.mean([statistics.stdev(s) for s in scores.values() if len(s) > 1] or [0.0])
        self.warnings = self.monitor.iteration(feedback.candidate_scores, game_sd, n,
                                               feedback.best_latest, best.mean, games.get(feedback.best_id),
                                               self.state["baseline"] if s["objective"] == "strength" else None)
        self.warnings += self.optimizer.warnings(self.space.names())

        seconds = time.monotonic() - started
        self.state["hours"] += seconds / 3600
        self.state["iterations"] += 1
        self.state["iteration_seconds"].append(seconds)

        record = {"iteration": self.state["iterations"], "objective": s["objective"], "rules": stage.rules, "sims": stage.sims, "seed0": seed0, "games": n,
                  "optimizer": self.optimizer.name,
                  "scores": {key: round(m, 5) for key, m in means.items()},
                  "best": {"index": best.index, "mean": best.mean, "evaluations": best.evaluations,
                           "values": self.space.values(best.unit)},
                  **self.optimizer.diagnostics(), "warnings": self.warnings, "seconds": seconds}
        self.append("evals.jsonl", record)
        _write_json(self.path("best.json"), self.config(best.unit))
        return best

    def holdout(self, best):
        started = time.monotonic()
        games = self.settings["holdout"]["games"]
        self.activity("holdout")
        scores, _ = self.runner.scores([("best", self.config(best.unit), self.stage(), HOLDOUT_SEEDS, games)])
        base = self.state["baseline"]["mean"]
        value = statistics.mean(scores["best"])
        self.account(started)
        self.log(f"holdout: best {value:.4f} vs baseline {base:.4f} ({value - base:+.4f})")
        self.state["holdouts"].append({"iteration": self.state["iterations"], "best": value, "baseline": base})
        if value < base and best.mean > base:
            self.warnings.append("holdout: the best beats the baseline on tuning seeds but not on held-out ones")

    def gate(self, best):
        stage = self.stage()
        settings = self.settings["gate"]
        candidate_values = self.space.values(best.unit)
        candidate = self.space.config(candidate_values, self.state["base_config"])
        release = self.space.config(self.state["release"]["values"], self.state["base_config"])
        if candidate == release:
            self.log("result: the best is the release; nothing to test", show=True)
            return
        self.log(f"gate at {stage.sims} sims, garbage x{stage.garbage_scale}, {stage.rules} rules")
        self.activity("release gate")
        started = time.monotonic()
        result = run_gate(self.runner, stage, candidate, release, self.state["gate_seed_next"],
                          delta=settings["delta"], alpha=settings["alpha"], beta=settings["beta"],
                          max_games=settings["max_games"], log=self.log)
        self.state["gate_seed_next"] = result["seed_end"]
        self.account(started)
        result.update({"iteration": self.state["iterations"], "objective": stage.objective, "rules": stage.rules, "hours": self.state["hours"], "values": candidate_values})
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
        plan = self.state["plan"]
        if not plan or self.state["phase"] == "done":
            return 0.0
        recent = self.state["iteration_seconds"][-3:]
        per = statistics.mean(recent) / 3600 if recent else plan["projected_hours"]["iteration"]
        return (self.settings["iterations"] - self.state["iterations"]) * per + plan["projected_hours"]["gate"]

    def eta(self, hours):
        finish = time.strftime("%a %H:%M", time.localtime(time.time() + hours * 3600))
        return f"done by {finish} at the latest ({hours:.1f} h left)"

    def optimizer_status(self):
        return ", ".join(f"{key} {value:.3f}" if isinstance(value, (int, float)) else f"{key} {value}"
                         for key, value in self.optimizer.diagnostics().items())

    def status(self, best):
        s, plan = self.settings, self.state["plan"]
        done, total = self.state["iterations"], s["iterations"]
        lines = [f"run {self.dir}  ({self.rules()} rules, {self.state['hours']:.2f} h, {self.runner.games_played} games this session, "
                 f"{self.runner.failures} game failures)"]
        lines.append(f"objective: {s['objective']}" + (" (sims/s)" if s["objective"] == "sims-per-second" else " (attack/piece)"))
        lines.append(f"{self.runner.threads} threads per game, {self.runner.workers} concurrent games")
        if plan:
            lines.append(f"{self.state['phase']}: {plan['sims']} sims, garbage x{plan['garbage_scale']}, iteration "
                         f"{done}/{total}" + ("" if self.state["phase"] == "done" else
                                             f", {self.eta(self.hours_left())}"))
        if best:
            base = (self.state["baseline"] or {}).get("mean")
            lines.append(f"best: {best.mean:.4f} over {best.evaluations} evaluations"
                         + (f" (baseline {base:.4f}, {best.mean - base:+.4f})" if base is not None else ""))
        if self.state["holdouts"]:
            h = self.state["holdouts"][-1]
            lines.append(f"holdout after iteration {h['iteration']}: {h['best']:.4f} vs baseline {h['baseline']:.4f} "
                         f"({h['best'] - h['baseline']:+.4f})")
        lines.append(f"release: {self.state['release']['source']} ({self.state['release']['gates']} gates passed)")
        lines.append(f"optimizer: {self.optimizer.name}; " + self.optimizer_status())
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
            self.status(self.optimizer.best())
            if plan_only:
                self.log("stopping after the plan as asked (resume to tune)", show=True)
                return
        total = self.settings["iterations"]
        every = self.settings["holdout"]["every"]
        if self.state["phase"] == "tune":
            self.baseline()
            while self.state["iterations"] < total:
                best = self.iterate()
                self.log(f"iteration {self.state['iterations']}/{total}: best {best.mean:.4f} ({best.evaluations} evals), "
                         f"{self.optimizer_status()}, {self.state['iteration_seconds'][-1] / 60:.1f} min")
                if self.state["iterations"] % every == 0 or self.state["iterations"] == total:
                    self.holdout(best)
                if not self.verbose:
                    self.log(f"iteration {self.state['iterations']}/{total}, {self.eta(self.hours_left())}", show=True)
                for w in self.warnings:
                    kind = w.split(":")[0]
                    self.log("WARNING " + w, show=kind not in self.warned)
                    self.warned.add(kind)
                self.save()
                self.status(best)
            self.state["phase"] = "gate"
            self.save()
        if self.state["phase"] == "gate":
            self.gate(self.optimizer.best())
            self.state["phase"] = "done"
            self.save()
            self.status(self.optimizer.best())
        self.log(f"done in {self.state['hours']:.2f} h; release.json is the result", show=True)
