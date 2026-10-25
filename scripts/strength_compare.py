#!/usr/bin/env python3
"""Compare two strength runs config by config.

    strength_compare.py --a DIR [DIR ...] --b DIR [DIR ...] [--a-name plain] [--b-name new]
"""
import argparse
import math
import pathlib
import random
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from strength_report import REPS, app, interval, load, parse, variant  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", nargs="+", required=True)
    parser.add_argument("--b", nargs="+", required=True)
    parser.add_argument("--a-name", default="A")
    parser.add_argument("--b-name", default="B")
    args = parser.parse_args()
    a, b = load(args.a), load(args.b)
    rng = random.Random(1)
    per_piece = lambda games: sum(g["attack"] for g in games) / max(sum(g["pieces"] for g in games), 1)
    app_mean = lambda games: statistics.fmean(app(g) for g in games)

    print(f"{args.b_name} / {args.a_name}, paired over the seeds both played; ratio of means [95% CI]")
    print(f"{'config':<20} {'seeds':>5} {args.a_name + ' APP@H':>12} {args.b_name + ' APP@H':>12}"
          f" {'APP@H ratio':>22} {'attack rate ratio':>22} {'deaths':>10} {'sims/move':>10}")
    names = sorted(set(a) & set(b), key=lambda n: (parse(n)[0], parse(n)[2] != 6, parse(n)[1], variant(n)))
    for name in names:
        seeds = sorted(set(a[name]) & set(b[name]))
        if not seeds:
            continue
        pairs = [(a[name][s], b[name][s]) for s in seeds]

        def ratio(statistic):
            def r(p):
                base = statistic([x for x, _ in p])
                return statistic([y for _, y in p]) / base if base else math.nan
            low, high = interval([r(rng.choices(pairs, k=len(pairs))) for _ in range(REPS)])
            return f"x{r(pairs):.3f} [{low:.3f}, {high:.3f}]"

        side_a, side_b = [x for x, _ in pairs], [y for _, y in pairs]
        sims = lambda games: statistics.fmean(g["sims_per_move"] for g in games)
        deaths = f"{sum(g['died'] for g in side_b)} vs {sum(g['died'] for g in side_a)}"
        print(f"{name:<20} {len(seeds):>5} {app_mean(side_a):>12.3f} {app_mean(side_b):>12.3f}"
              f" {ratio(app_mean):>22} {ratio(per_piece):>22} {deaths:>10} {f'x{sims(side_b) / sims(side_a):.3f}':>10}")


if __name__ == "__main__":
    main()
