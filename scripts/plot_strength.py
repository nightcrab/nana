#!/usr/bin/env python3
"""Plot a strength_ladder.sh / strength_batches.sh run for the README, light and dark.

    plot_strength.py DIR [DIR ...] [--out docs]
"""
import argparse
import math
import pathlib
import random
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from strength_report import REPS, app, crossing, interval, line_fit, load, parse  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", secondary="#52514e", muted="#898781", grid="#e1e0d9",
                  s1="#2a78d6", s2="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", secondary="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 s1="#3987e5", s2="#d95926"),
}


def font():
    for name in ("Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "sans-serif"


def analyse(configs):
    ladder = sorted((n for n in configs if parse(n)[0] == 1 and parse(n)[2] == 6), key=lambda n: parse(n)[1])
    parallel = sorted((n for n in configs if parse(n)[0] > 1), key=lambda n: parse(n)[0])
    seeds = sorted(set.intersection(*(set(configs[n]) for n in ladder + parallel)))
    sims = lambda g: g["sims_per_move"]

    def estimate(sample):
        mean = lambda n, f: statistics.fmean(f(configs[n][s]) for s in sample)
        a, b = line_fit([(math.log2(parse(n)[1]), math.log(mean(n, app))) for n in ladder])
        lad_sims = [(parse(n)[1], mean(n, sims)) for n in ladder]
        return {n: ((math.log(mean(n, app)) - a) / b, crossing(lad_sims, mean(n, sims))) for n in parallel}

    rng = random.Random(1)
    point = estimate(seeds)
    boot = [estimate(rng.choices(seeds, k=len(seeds))) for _ in range(REPS)]
    rows = []
    for n in parallel:
        lo, hi = interval([o[n][0] for o in boot])
        rows.append(dict(threads=parse(n)[0], eff=2 ** point[n][0], eff_lo=2 ** lo, eff_hi=2 ** hi,
                         sims=2 ** point[n][1]))

    def series(names):
        out = []
        for n in names:
            values = [app(configs[n][s]) for s in seeds]
            lo, hi = interval([statistics.fmean(rng.choices(values, k=len(values))) for _ in range(REPS)])
            out.append(dict(name=n, sims=statistics.fmean(sims(configs[n][s]) for s in seeds),
                            app=statistics.fmean(values), lo=lo, hi=hi))
        return out

    base = configs[ladder[0]][seeds[0]]
    meta = dict(games=len(seeds), base_ms=base["base_ms"], horizon=base["max_pieces"])
    return series(ladder), series([ladder[0]] + parallel), rows, meta


def style(ax, t, f):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], labelcolor=t["muted"], labelsize=12, length=0, pad=8)
    ax.grid(True, axis="y", color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(f)


def header(fig, t, f, title, subtitle, legend):
    fig.text(0.031, 0.93, title, fontsize=20, fontweight="semibold", color=t["ink"], fontfamily=f, va="top")
    fig.text(0.031, 0.855, subtitle, fontsize=13, color=t["secondary"], fontfamily=f, va="top")
    x = 0.031
    for text, colour, dashed in legend:
        fig.add_artist(plt.Line2D([x, x + 0.014], [0.775, 0.775], color=colour, linewidth=3,
                                  linestyle=(0, (3, 2)) if dashed else "-", transform=fig.transFigure))
        label = fig.text(x + 0.02, 0.775, text, fontsize=13, color=t["ink"], fontfamily=f, va="center")
        fig.canvas.draw()
        box = label.get_window_extent().transformed(fig.transFigure.inverted())
        x = box.x1 + 0.025


def log2_axis(ax, ticks, fmt, t, f, axis="x"):
    getattr(ax, f"set_{axis}scale")("log", base=2)
    a = getattr(ax, f"{axis}axis")
    a.set_major_locator(FixedLocator(ticks))
    a.set_minor_locator(NullLocator())
    a.set_major_formatter(FuncFormatter(lambda v, _: fmt(v)))


def human(v):
    return f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"


def plot_effort(path, t, f, ladder, parallel, meta):
    fig = plt.figure(figsize=(2048 / 150, 980 / 150), dpi=150, facecolor=t["surface"])
    ax = fig.add_axes([0.075, 0.12, 0.84, 0.58])
    style(ax, t, f)
    header(fig, t, f, "nana playing strength: search effort",
           f"Attack per piece over the first {meta['horizon']} pieces against the emulated opponent, "
           f"{meta['games']} paired games per point · AMD EPYC 9R45",
           [(f"1 thread, {meta['base_ms']} to {meta['base_ms'] * 32} ms per move", t["s1"], False),
            (f"1 to 32 threads, {meta['base_ms']} ms per move", t["s2"], False)])
    for data, colour, labels in ((ladder, t["s1"], None), (parallel, t["s2"], "threads")):
        xs = [d["sims"] for d in data]
        ax.fill_between(xs, [d["lo"] for d in data], [d["hi"] for d in data], color=colour, alpha=0.14, linewidth=0)
        ax.plot(xs, [d["app"] for d in data], color=colour, linewidth=2, zorder=3)
        ax.scatter(xs, [d["app"] for d in data], s=64, color=colour, edgecolors=t["surface"], linewidths=2, zorder=4)
    def other_at(data, x):
        for a, b in zip(data, data[1:]):
            if a["sims"] <= x <= b["sims"]:
                w = math.log(x / a["sims"]) / math.log(b["sims"] / a["sims"])
                return a["app"] + w * (b["app"] - a["app"])
        return None

    for data, other, text in ((ladder, parallel, lambda d: f"{parse(d['name'])[1]}x"),
                              (parallel, ladder, lambda d: f"{parse(d['name'])[0]} threads")):
        for d in data[1:]:
            y = other_at(other, d["sims"])
            above = y is None or d["app"] >= y
            ax.annotate(text(d), (d["sims"], d["app"]), xytext=(0, 12 if above else -14), textcoords="offset points",
                        ha="center", va="bottom" if above else "top", fontsize=11, color=t["secondary"], fontfamily=f)
    ticks = [6000 * 2 ** i for i in range(6)]
    log2_axis(ax, [round(v, -3) for v in ticks], human, t, f)
    ax.set_xlim(ladder[0]["sims"] / 1.3, ladder[-1]["sims"] * 1.3)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_xlabel("simulations per move", color=t["muted"], fontsize=12, fontfamily=f)
    ax.set_ylabel("attack per piece", color=t["muted"], fontsize=12, fontfamily=f)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(f)
    fig.savefig(path, dpi=150, facecolor=t["surface"])
    plt.close(fig)


def plot_speedup(path, t, f, rows, meta):
    fig = plt.figure(figsize=(2048 / 150, 980 / 150), dpi=150, facecolor=t["surface"])
    ax = fig.add_axes([0.075, 0.12, 0.84, 0.58])
    style(ax, t, f)
    header(fig, t, f, "nana playing strength: speedup over 1 thread",
           f"Strength speedup: the 1-thread think time that plays as well, from a straight-line fit of the "
           f"1-thread ladder · 95% bootstrap band · {meta['games']} games per point",
           [("ideal", t["muted"], True), ("simulations per move", t["s2"], False), ("strength", t["s1"], False)])
    threads = [1] + [r["threads"] for r in rows]
    ax.plot(threads, threads, color=t["muted"], linewidth=1.5, linestyle=(0, (4, 3)), zorder=2)
    sims = [1] + [r["sims"] for r in rows]
    ax.plot(threads, sims, color=t["s2"], linewidth=2, zorder=3)
    ax.scatter(threads, sims, s=64, color=t["s2"], edgecolors=t["surface"], linewidths=2, zorder=4)
    xs = [r["threads"] for r in rows]
    ax.fill_between(xs, [r["eff_lo"] for r in rows], [r["eff_hi"] for r in rows], color=t["s1"], alpha=0.16,
                    linewidth=0)
    ax.plot(xs, [r["eff"] for r in rows], color=t["s1"], linewidth=2, zorder=3)
    ax.scatter(xs, [r["eff"] for r in rows], s=64, color=t["s1"], edgecolors=t["surface"], linewidths=2, zorder=4)
    last = rows[-1]
    for value, colour, text in ((last["sims"], t["s2"], f"{last['sims']:.1f}x simulations"),
                                (last["eff"], t["s1"], f"{last['eff']:.1f}x strength")):
        ax.annotate(text, (last["threads"], value), xytext=(10, 0), textcoords="offset points", va="center",
                    fontsize=12, color=t["ink"], fontweight="semibold", fontfamily=f)
    log2_axis(ax, [1, 2, 4, 8, 16, 32], lambda v: f"{v:g}", t, f)
    log2_axis(ax, [1, 2, 4, 8, 16, 32], lambda v: f"{v:g}x", t, f, axis="y")
    ax.set_xlim(0.85, 48)
    ax.set_ylim(0.85, 36)
    ax.set_xlabel("threads", color=t["muted"], fontsize=12, fontfamily=f)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(f)
    fig.savefig(path, dpi=150, facecolor=t["surface"])
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs"))
    args = parser.parse_args()
    ladder, parallel, rows, meta = analyse(load(args.dirs))
    f = font()
    for name, t in THEMES.items():
        plot_effort(args.out / f"strength-effort-{name}.png", t, f, ladder, parallel, meta)
        plot_speedup(args.out / f"strength-speedup-{name}.png", t, f, rows, meta)
    for r in rows:
        print(f"{r['threads']:>2} threads: strength {r['eff']:.2f}x [{r['eff_lo']:.2f}, {r['eff_hi']:.2f}],"
              f" simulations {r['sims']:.2f}x, efficiency {100 * r['eff'] / r['sims']:.0f}%")


if __name__ == "__main__":
    main()
