"""nana's autotuner. See tuner/README.md.

    python -m tuner start NAME --nana PATH --cpus 0,2,4,6 (--budget-hours H | --sims N)
                               [--space tuner/spaces/eval.json ...] [--iterations 15] [--particles 24]
                               [--games 32] [--spread 0.3] [--gate-games 400] [--threads 1] [--plan-only]
    python -m tuner resume NAME
    python -m tuner status NAME
    python -m tuner check NAME          health checks only
"""
import argparse
import json
import os
import shutil
import sys

from .health import HealthError
from .run import Run
from .runner import Runner

HERE = os.path.dirname(os.path.abspath(__file__))


def run_dir(name):
    return name if os.path.sep in name or "/" in name else os.path.join(HERE, "runs", name)


def describe(directory):
    try:
        with open(os.path.join(directory, "state.json")) as f:
            state = json.load(f)
        with open(os.path.join(directory, "run.json")) as f:
            settings = json.load(f)
    except (OSError, ValueError):
        return "never started"
    release = state.get("release", {})
    line = f"{state.get('phase', '?')}, iteration {state.get('iterations', 0)}/{settings.get('iterations', '?')}"
    if release.get("gates"):
        line += ", with a release that passed its gate"
    return line


def clear_existing(directory, overwrite):
    if not os.path.exists(os.path.join(directory, "run.json")):
        return
    question = f"{directory} already holds a run ({describe(directory)}). Overwrite it? [y/N] "
    if not overwrite:
        if not sys.stdin.isatty():
            raise SystemExit(f"{directory} already holds a run; pass --overwrite, use resume, or another name")
        try:
            answer = input(question)
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            raise SystemExit("\nkept it; use resume to continue it, or another name")
    shutil.rmtree(directory)


def main():
    parser = argparse.ArgumentParser(prog="python -m tuner", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="create a run, calibrate it and start it")
    start.add_argument("name")
    start.add_argument("--nana", required=True, help="nana_emulate built in Release")
    start.add_argument("--cpus", required=True, help="logical cpus to use, e.g. 0,2,4,6 (one per physical core)")
    start.add_argument("--threads", type=int, default=1, help="search threads per game (default 1; must divide the CPU count)")
    budget = start.add_mutually_exclusive_group(required=True)
    budget.add_argument("--budget-hours", type=float,
                        help="the whole run's time: the tuner picks the largest sims count that fits")
    budget.add_argument("--sims", type=int, help="tune at this sims count, whatever it costs")
    start.add_argument("--space", action="append", help="parameter space files (default tuner/spaces/eval.json)")
    start.add_argument("--objective", choices=["strength", "sims-per-second"], default="strength",
                       help="fitness to maximise (default strength; sims-per-second defaults to search.json)")
    start.add_argument("--iterations", type=int, help="PSO iterations (default 15)")
    start.add_argument("--particles", type=int, help="default 24")
    start.add_argument("--games", dest="games_per_eval", type=int, help="games per evaluation (strength: 32; sims-per-second: 4)")
    start.add_argument("--spread", type=float,
                       help="the starting swarm's spread around the defaults, in unit-cube units (default 0.3)")
    start.add_argument("--gate-games", type=int, help="the release gate's limit on paired games (strength: 400; sims-per-second: 32)")
    start.add_argument("--holdout-every", type=int, help="iterations between holdouts (default 5)")
    start.add_argument("--chunk", type=int, help="games per nana_emulate process (default 4)")
    start.add_argument("--seed", type=int, help="the optimiser's seed (default 1)")
    start.add_argument("--rules", choices=["tetrio", "botris"], help="the ruleset games are played with (default tetrio)")
    start.add_argument("--eval", choices=["nana", "convex"], help="evaluator style to tune (default: nana_emulate's configured default)")
    start.add_argument("--plan-only", action="store_true", help="calibrate, print the plan and stop; resume tunes")
    start.add_argument("-v", "--verbose", action="store_true", help="print everything (log.txt always has it)")
    start.add_argument("--overwrite", action="store_true", help="replace a run of the same name without asking")

    resume = sub.add_parser("resume", help="continue a run from its checkpoint")
    resume.add_argument("name")
    resume.add_argument("-v", "--verbose", action="store_true", help="print everything (log.txt always has it)")

    for name in ("status", "check"):
        p = sub.add_parser(name)
        p.add_argument("name")

    args = parser.parse_args()
    directory = run_dir(args.name)

    try:
        if args.command == "start":
            cpus = [int(c) for c in args.cpus.split(",")]
            try:
                Runner.validate_cpus(cpus, args.threads)
            except ValueError as e:
                parser.error(str(e))
            spaces = args.space or [os.path.join(HERE, "spaces",
                                                "search.json" if args.objective == "sims-per-second" else "eval.json")]
            options = {k: getattr(args, k) for k in ("budget_hours", "sims", "iterations", "particles", "games_per_eval",
                                                     "spread", "gate_games", "holdout_every", "chunk", "seed", "rules", "objective", "threads", "eval")}
            if not os.path.isfile(args.nana):
                raise SystemExit(f"no nana_emulate at {args.nana}")
            clear_existing(directory, args.overwrite)
            run = Run.create(directory, args.nana, cpus, spaces, options, args.verbose)
            run.run(plan_only=args.plan_only)
        elif args.command == "resume":
            Run(directory, args.verbose).run()
        elif args.command == "status":
            with open(os.path.join(directory, "status.txt")) as f:
                print(f.read(), end="")
        elif args.command == "check":
            Run(directory).health_checks()
    except HealthError as e:
        print(f"health check failed: {e}", file=sys.stderr)
        sys.exit(3)
    except KeyboardInterrupt:
        print("\ninterrupted; the last iteration's checkpoint is saved, resume to continue", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
