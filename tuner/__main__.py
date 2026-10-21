"""nana's autotuner. See tuner/README.md.

    python -m tuner start NAME --nana PATH --cpus 0,2,4,6 [--space tuner/spaces/eval.json ...]
                               [--schedule tuner/schedules/week.json] [--particles 24]
                               [--total-hours H | --time-scale X]
    python -m tuner resume NAME [--hours H] [--total-hours H | --time-scale X]
    python -m tuner restart-stage NAME K [--from release|best|CONFIG.json]
    python -m tuner add-params NAME SPACE.json
    python -m tuner status NAME
    python -m tuner check NAME          health checks only
"""

import argparse
import os
import sys

from .health import HealthError
from .run import Run

HERE = os.path.dirname(os.path.abspath(__file__))


def run_dir(name):
    return name if os.path.sep in name or "/" in name else os.path.join(HERE, "runs", name)


def add_timeline(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--total-hours", type=float,
                       help="stretch or squeeze the schedule so its stages add up to this many hours")
    group.add_argument("--time-scale", type=float, help="multiply every stage's hours (and the gate interval)")


def apply_timeline(run, args):
    if args.total_hours is not None or args.time_scale is not None:
        run.set_time_scale(scale=args.time_scale, total_hours=args.total_hours)


def main():
    parser = argparse.ArgumentParser(prog="python -m tuner", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="create a run and start it")
    start.add_argument("name")
    start.add_argument("--nana", required=True, help="nana_play built in Release")
    start.add_argument("--cpus", required=True, help="logical cpus to use, e.g. 0,2,4,6 (one per physical core)")
    start.add_argument("--space", action="append", help="parameter space files (default tuner/spaces/eval.json)")
    start.add_argument("--schedule", default=os.path.join(HERE, "schedules", "week.json"))
    start.add_argument("--particles", type=int, default=24)
    start.add_argument("--chunk", type=int, default=4, help="games per nana_play process")
    start.add_argument("--seed", type=int, default=1)
    start.add_argument("--hours", type=float, help="stop after this many hours (resume continues)")
    add_timeline(start)

    resume = sub.add_parser("resume", help="continue a run from its checkpoint")
    resume.add_argument("name")
    resume.add_argument("--hours", type=float)
    add_timeline(resume)

    restart = sub.add_parser("restart-stage", help="rerun the ramp from stage K, warm-started from a config")
    restart.add_argument("name")
    restart.add_argument("stage", type=int)
    restart.add_argument("--from", dest="source", default="release",
                         help="release (default), best, or a NANA_CONFIG file")

    add = sub.add_parser("add-params", help="add parameters to a run's space")
    add.add_argument("name")
    add.add_argument("space")

    for name in ("status", "check"):
        p = sub.add_parser(name)
        p.add_argument("name")

    args = parser.parse_args()
    directory = run_dir(args.name)

    try:
        if args.command == "start":
            cpus = [int(c) for c in args.cpus.split(",")]
            spaces = args.space or [os.path.join(HERE, "spaces", "eval.json")]
            run = Run.create(directory, args.nana, cpus, spaces, args.schedule, args.particles, args.chunk, args.seed)
            apply_timeline(run, args)
            run.run(args.hours)
        elif args.command == "resume":
            run = Run(directory)
            apply_timeline(run, args)
            run.run(args.hours)
        elif args.command == "restart-stage":
            Run(directory).restart_stage(args.stage, args.source)
        elif args.command == "add-params":
            Run(directory).add_params(args.space)
        elif args.command == "status":
            with open(os.path.join(directory, "status.txt")) as f:
                print(f.read(), end="")
        elif args.command == "check":
            Run(directory).health_checks()
    except HealthError as e:
        print(f"health check failed: {e}", file=sys.stderr)
        sys.exit(3)
    except KeyboardInterrupt:
        print("interrupted; the last iteration's checkpoint is saved, resume to continue", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
