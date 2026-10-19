"""Plot thread scaling from a bench_scaling.sh runs.csv.

usage: python scripts/plot_scaling.py <runs.csv or results dir> [out.png] [--builds mp,tds] [--linear]
  --builds  which builds to plot (default mp; TDS-UCT is root-bound and no longer benchmarked by default)
  --linear  linear axes from 0 instead of log2, so efficiency loss shows as the gap to the dashed line

One column per position test, rows for simulations/s and nodes/s. Points are the
median of reps, whiskers the min-max. Dotted lines are linear scaling from each build's
own 1-thread median. Dashed lines, when the sweep ran with TARGET=1, are the measured
target: N x the rate of 1 thread searching N times as long, which grows the same size of
tree and so accounts for simulations slowing as the tree grows. The label at the last
point is its speedup and, with a target, the share of the target reached. By default
both axes are log2, so a straight line at the dotted slope is perfect scaling.
Nodes/s is not comparable between builds (MP-MCTS restarts skip select hops).
"""
import argparse
import collections
import csv
import pathlib
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullLocator

BUILDS = {"tds": ("TDS-UCT", "#2a78d6"), "mp": ("MP-MCTS", "#eb6834")}
METRICS = [("sims_per_s", "simulations / s"), ("nodes_per_s", "nodes / s")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def human(v, _=None):
    for div, suffix in ((1e6, "M"), (1e3, "k")):
        if v >= div:
            return f"{v / div:g}{suffix}"
    return f"{v:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path, nargs="?")
    ap.add_argument("--builds", default="mp")
    ap.add_argument("--linear", action="store_true")
    args = ap.parse_args()
    builds = {b: BUILDS[b] for b in args.builds.split(",")}
    src = args.src
    csv_path = next(src.rglob("runs.csv")) if src.is_dir() else src
    out = args.out or csv_path.with_name("scaling-linear.png" if args.linear else "scaling.png")

    rows = list(csv.DictReader(open(csv_path)))
    # sweeps before TARGET have no ms column; otherwise the scaling runs share the shortest ms
    has_ms = rows and "ms" in rows[0]
    base_ms = min(int(r["ms"]) for r in rows) if has_ms else None
    data = collections.defaultdict(list)
    long_runs = collections.defaultdict(list)
    for r in rows:
        if has_ms and int(r["ms"]) != base_ms:
            long_runs[(r["build"], r["test"], int(r["ms"]) // base_ms)].append(r)
        else:
            data[(r["build"], r["test"], int(r["threads"]))].append(r)
    tests = sorted({k[1] for k in data})
    threads = sorted({k[2] for k in data})
    machine = csv_path.with_name("machine.txt")
    title = machine.read_text().splitlines()[0] if machine.exists() else csv_path.parent.name

    fig, axes = plt.subplots(len(METRICS), len(tests), figsize=(5.2 * len(tests), 4.2 * len(METRICS)), squeeze=False)
    for col, test in enumerate(tests):
        for row, (key, label) in enumerate(METRICS):
            ax = axes[row][col]
            ends = []
            for build, (name, color) in builds.items():
                ns = [n for n in threads if data.get((build, test, n))]
                if not ns:
                    continue
                vals = [[float(r[key]) for r in data[(build, test, n)] if r[key]] for n in ns]
                med = [statistics.median(v) for v in vals]
                lo = [m - min(v) for m, v in zip(med, vals)]
                hi = [max(v) - m for m, v in zip(med, vals)]
                ax.plot(ns, [med[0] * n / ns[0] for n in ns], color=color, lw=1, ls=(0, (1, 2)), alpha=0.45)
                target = {n: n * statistics.median(float(r[key]) for r in long_runs[(build, test, n)] if r[key])
                          for n in ns if long_runs.get((build, test, n))}
                if target:
                    target[ns[0]] = med[0]
                    tn = sorted(target)
                    ax.plot(tn, [target[n] for n in tn], color=color, lw=1.2, ls=(0, (4, 3)), alpha=0.7)
                ax.errorbar(ns, med, yerr=[lo, hi], color=color, lw=2, marker="o", ms=6,
                            mec="white", mew=1.5, capsize=0, elinewidth=1.5, label=name)
                speedup = f"{med[-1] / med[0]:.1f}x"
                if ns[-1] in target and ns[-1] != ns[0]:
                    speedup += f", {100 * med[-1] / target[ns[-1]]:.0f}% of target"
                ends.append((med[-1], ns[-1], speedup if len(builds) == 1 else f"{name.split('-')[0]} {speedup}"))
            # speedup labels, nudged apart in endpoint order so coincident endpoints stay readable
            for i, (y, x, text) in enumerate(sorted(ends)):
                dy = 0 if len(ends) == 1 else (12 * i / (len(ends) - 1) - 6)
                ax.annotate(text, (x, y), xytext=(7, dy), textcoords="offset points", va="center",
                            fontsize=9, color=INK)
            if args.linear:
                ax.set_xlim(0, threads[-1] * 1.45)
                ax.set_ylim(bottom=0)
            else:
                ax.set_xscale("log", base=2)
                ax.set_yscale("log", base=2)
                ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
                ax.yaxis.set_minor_locator(NullLocator())
                ax.set_xlim(threads[0] / 1.3, threads[-1] * 4)
            ax.xaxis.set_major_locator(FixedLocator(threads))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_major_formatter(FuncFormatter(human))
            ax.grid(True, color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelsize=9)
            ax.set_title(f"test {test}: {label}", fontsize=11, color=INK, loc="left")
            ax.set_xlabel("threads", fontsize=9, color=MUTED)
            if row == 0 and col == 0 and len(builds) > 1:
                ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    who = next(iter(builds.values()))[0] + "  " if len(builds) == 1 else ""
    lines = "dotted = linear" + (", dashed = target (1 thread run N times as long)" if long_runs else "")
    fig.suptitle(who + title + "\nmedian of reps, min-max whiskers, " + lines, fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="#fcfcfb")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
