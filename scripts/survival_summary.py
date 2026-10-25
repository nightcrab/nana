#!/usr/bin/env python3
"""Summarise nana_emulate survival runs.

    survival_summary.py RUN.jsonl [RUN.jsonl ...]
    survival_summary.py --paired A.jsonl B.jsonl
"""
import argparse
import json
import random
import statistics


def load(path):
    games = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                game = json.loads(line)
                games[game["seed"]] = game
    return games


def bootstrap(values, statistic, reps=4000, seed=1):
    rng = random.Random(seed)
    n = len(values)
    samples = sorted(statistic([values[rng.randrange(n)] for _ in range(n)]) for _ in range(reps))
    return samples[int(0.025 * reps)], samples[int(0.975 * reps)]


def mean_pieces(games):
    return statistics.fmean(g["pieces"] for g in games)


def describe(path):
    games = list(load(path).values())
    if not games:
        print(f"{path}: no games")
        return
    pieces = [g["pieces"] for g in games]
    deaths = sum(g["died"] for g in games)
    low, high = bootstrap(games, mean_pieces)
    first = games[0]
    line = (f"{path}: threads {first['threads']} ms {first['ms']} ramp {first['ramp']}"
            f" | games {len(games)} died {deaths}"
            f" | mean {statistics.fmean(pieces):.1f} [{low:.1f}, {high:.1f}]"
            f" median {statistics.median(pieces):.0f}"
            f" cv {statistics.pstdev(pieces) / statistics.fmean(pieces):.2f}")
    if deaths < len(games):
        line += f" | pieces/death {sum(pieces) / max(deaths, 1):.1f}"
    line += f" | app {sum(g['attack'] for g in games) / max(sum(pieces), 1):.3f}"
    line += (f" | sims/move {statistics.fmean(g['sims_per_move'] for g in games):.0f}"
             f" ms/move {statistics.fmean(g['ms_per_move'] for g in games):.1f}")
    if "stop_ms" in first:
        line += " (start {:.1f} oversleep {:.1f} stop {:.1f})".format(
            *(statistics.fmean(g[key] for g in games) for key in ("start_ms", "oversleep_ms", "stop_ms")))
    print(line)


def paired(path_a, path_b):
    a, b = load(path_a), load(path_b)
    seeds = sorted(set(a) & set(b))
    if not seeds:
        print("no common seeds")
        return
    pairs = [(a[s]["pieces"], b[s]["pieces"]) for s in seeds]

    def ratio(sample):
        return sum(y for _, y in sample) / max(sum(x for x, _ in sample), 1)

    low, high = bootstrap(pairs, ratio)
    wins = sum(y > x for x, y in pairs)
    ties = sum(y == x for x, y in pairs)
    print(f"{len(seeds)} common seeds: B/A mean survival x{ratio(pairs):.3f} [{low:.3f}, {high:.3f}],"
          f" B survived longer on {wins}, tied {ties}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paired", action="store_true", help="compare exactly two runs seed by seed")
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()

    if args.paired:
        if len(args.runs) != 2:
            parser.error("--paired takes two runs")
        for path in args.runs:
            describe(path)
        paired(*args.runs)
    else:
        for path in args.runs:
            describe(path)


if __name__ == "__main__":
    main()
