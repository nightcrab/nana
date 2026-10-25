#!/usr/bin/env python3
"""Report a strength_ladder.sh run.

    strength_report.py DIR [DIR ...] [--plot out.png] [--dark]
"""
import argparse
import collections
import json
import math
import pathlib
import random
import re
import statistics

BASELINE = "t1-x1-lf6"
REPS = 2000


def load(dirs):
    configs = collections.defaultdict(dict)
    for d in dirs:
        for path in sorted(pathlib.Path(d).glob("t*-x*-lf*.jsonl")):
            for line in open(path, encoding="utf-8-sig"):
                if line.startswith("{"):
                    game = json.loads(line)
                    configs[game["config"]][game["seed"]] = game
    return configs


NAME = re.compile(r"t(\d+)-x(\d+)-lf(\d+)(?:-v([\w.-]+))?")


def parse(name):
    match = NAME.fullmatch(name)
    return int(match[1]), int(match[2]), int(match[3])


def variant(name):
    return NAME.fullmatch(name)[4] or ""


def app(game):
    return game["attack"] / game["max_pieces"]


def true_app(game):
    return game["true_attack"] / game["max_pieces"]


def interval(samples):
    finite = sorted(s for s in samples if not math.isnan(s))
    if not finite:
        return math.nan, math.nan
    return finite[int(0.025 * len(finite))], finite[min(int(0.975 * len(finite)), len(finite) - 1)]


def crossing(ladder, value):
    if value <= ladder[0][1]:
        return -math.inf if value < ladder[0][1] else math.log2(ladder[0][0])
    for (m0, y0), (m1, y1) in zip(ladder, ladder[1:]):
        if y0 < value <= y1:
            return math.log2(m0) + (value - y0) / (y1 - y0) * (math.log2(m1) - math.log2(m0))
    return math.inf


def show_multiple(log2m, top):
    if math.isnan(log2m):
        return "n/a"
    if log2m == math.inf:
        return f">{top:g}x"
    if log2m == -math.inf:
        return "<1x"
    return f"{2 ** log2m:.2f}x"


def pct(part, whole):
    return part / whole if whole else math.nan


def show_ratio(log2r):
    return "n/a" if not math.isfinite(log2r) else f"{100 * 2 ** log2r:.0f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--plot", type=pathlib.Path)
    parser.add_argument("--dark", action="store_true", help="plot for a dark background")
    args = parser.parse_args()

    configs = load(args.dirs)
    if not configs:
        print("no games")
        return
    names = sorted(configs, key=lambda n: (parse(n)[0], parse(n)[2] != 6, parse(n)[1]))
    ladder_names = sorted((n for n in names if parse(n)[0] == 1 and parse(n)[2] == 6 and not variant(n)),
                          key=lambda n: parse(n)[1])
    parallel = [n for n in names if parse(n)[0] > 1]
    first = next(iter(configs[names[0]].values()))
    base_ms, horizon = first["base_ms"], first["max_pieces"]
    rng = random.Random(1)

    print(f"base think time {base_ms} ms, horizon {horizon} pieces")
    print("APP@H = attack / H (dying early sends nothing for the rest); 95% bootstrap CIs")
    print(f"{'config':<12} {'ms':>5} {'games':>5} {'APP@H':>6} {'95% CI':>15} {'±%':>5} {'cv':>5}"
          f" {'true APP@H':>10} {'reached H':>9} {'sims/move':>10} {'ms/move':>8}")
    stats = {}
    for name in names:
        games = list(configs[name].values())
        values = [app(g) for g in games]
        mean = statistics.fmean(values)
        low, high = interval([statistics.fmean(rng.choices(values, k=len(values))) for _ in range(REPS)])
        stats[name] = dict(values=values, mean=mean, low=low, high=high)
        print(f"{name:<12} {base_ms * parse(name)[1]:>5} {len(games):>5} {mean:>6.3f} [{low:>6.3f}, {high:>6.3f}]"
              f" {pct(50 * (high - low), mean):>5.1f} {pct(statistics.pstdev(values), mean):>5.2f}"
              f" {statistics.fmean(true_app(g) for g in games):>10.3f}"
              f" {sum(not g['died'] for g in games):>4}/{len(games):<4}"
              f" {statistics.fmean(g['sims_per_move'] for g in games):>10.0f}"
              f" {statistics.fmean(g['ms_per_move'] for g in games):>8.1f}")

    def paired(a, b, metric):
        seeds = sorted(set(configs[a]) & set(configs[b]))
        pairs = [(metric(configs[a][s]), metric(configs[b][s])) for s in seeds]
        ratio = lambda p: sum(y for _, y in p) / sum(x for x, _ in p) if sum(x for x, _ in p) else math.nan
        low, high = interval([ratio(rng.choices(pairs, k=len(pairs))) for _ in range(REPS)])
        return ratio(pairs), low, high, len(seeds), sum(y > x for x, y in pairs)

    variants_report(configs, names, rng)

    if BASELINE in configs:
        print(f"\npaired against {BASELINE}: ratio of means [95% CI] on common seeds")
        print(f"{'config':<12} {'APP@H':>22} {'true APP@H':>22} {'better on':>10}")
        for name in names:
            if name == BASELINE:
                continue
            r, lo, hi, n, wins = paired(BASELINE, name, app)
            tr, tlo, thi, _, _ = paired(BASELINE, name, true_app)
            print(f"{name:<12} {f'x{r:.3f} [{lo:.3f}, {hi:.3f}]':>22} {f'x{tr:.3f} [{tlo:.3f}, {thi:.3f}]':>22}"
                  f" {f'{wins}/{n}':>10}")

        print(f"\nbreakdown, paired against {BASELINE}: attack per piece actually played, and deaths before H")
        print(f"{'config':<12} {'attack rate':>22} {'deaths':>8} {'deaths, baseline':>17} {'died in both':>13}")
        base = configs[BASELINE]
        for name in names:
            if name == BASELINE:
                continue
            seeds = sorted(set(base) & set(configs[name]))
            pairs = [(base[s], configs[name][s]) for s in seeds]
            rate = lambda side: lambda p: (sum(g[side]["attack"] for g in p) / max(sum(g[side]["pieces"] for g in p), 1))
            ratio = lambda p: rate(1)(p) / rate(0)(p) if rate(0)(p) else math.nan
            low, high = interval([ratio(rng.choices(pairs, k=len(pairs))) for _ in range(REPS)])
            print(f"{name:<12} {f'x{ratio(pairs):.3f} [{low:.3f}, {high:.3f}]':>22}"
                  f" {sum(b['died'] for _, b in pairs):>8} {sum(a['died'] for a, _ in pairs):>17}"
                  f" {sum(a['died'] and b['died'] for a, b in pairs):>13}")

    results = {}
    if len(ladder_names) >= 3 and parallel:
        fit_report(configs, ladder_names, parallel, rng)

    if len(ladder_names) >= 2 and parallel:
        top = parse(ladder_names[-1])[1]
        print("\nparallel configs on the 1-thread ladder (APP@H; paired bootstrap over common seeds)")
        print(f"{'config':<12} {'effective':>24} {'sims multiple':>24} {'strength efficiency':>22}")
        for name in parallel:
            seeds = sorted(set.intersection(*(set(configs[n]) for n in ladder_names + [name])))
            if not seeds:
                continue

            def estimate(sample):
                mean = lambda n, f: statistics.fmean(f(configs[n][s]) for s in sample)
                sims = lambda g: g["sims_per_move"]
                lad = [(parse(n)[1], mean(n, app)) for n in ladder_names]
                lad_sims = [(parse(n)[1], mean(n, sims)) for n in ladder_names]
                eff, sim = crossing(lad, mean(name, app)), crossing(lad_sims, mean(name, sims))
                return eff, sim, (eff - sim) if math.isfinite(eff) and math.isfinite(sim) else math.nan

            point = estimate(seeds)
            boot = [estimate(rng.choices(seeds, k=len(seeds))) for _ in range(REPS)]
            cis = [interval([b[i] for b in boot]) for i in range(3)]
            results[name] = point
            fmt = lambda v, ci: f"{show_multiple(v, top)} [{show_multiple(ci[0], top)}, {show_multiple(ci[1], top)}]"
            print(f"{name:<12} {fmt(point[0], cis[0]):>24} {fmt(point[1], cis[1]):>24}"
                  f" {f'{show_ratio(point[2])} [{show_ratio(cis[2][0])}, {show_ratio(cis[2][1])}]':>22}"
                  f"  ({len(seeds)} seeds)")
        print("(efficiency = effective / sims multiple; intervals with > or < bounds left the ladder)")

    if args.plot:
        plot(args.plot, args.dark, base_ms, horizon, names, ladder_names, parallel, stats, results)
        print(f"\nplot: {args.plot}")


def variants_report(configs, names, rng):
    groups = {}
    for name in names:
        if variant(name):
            groups.setdefault(parse(name), []).append(name)
    for key, group in sorted(groups.items()):
        plain = f"t{key[0]}-x{key[1]}-lf{key[2]}"
        control = next((n for n in group if variant(n) in ("default", "off")), plain if plain in configs else None)
        if control is None:
            continue
        print(f"\nvariants at {key[0]} threads, paired against {control}: ratio of means [95% CI]")
        print(f"{'config':<20} {'seeds':>5} {'APP@H':>22} {'attack rate':>22} {'deaths':>11} {'sims/move':>10}"
              f" {'restarts':>9} {'to root':>8} {'leaks':>7}")
        mean_of = lambda games, k: statistics.fmean(g.get(k, math.nan) for g in games)
        per_piece = lambda games: sum(g["attack"] for g in games) / max(sum(g["pieces"] for g in games), 1)
        for name in [control] + [n for n in group if n != control]:
            seeds = sorted(set(configs[control]) & set(configs[name]))
            pairs = [(configs[control][s], configs[name][s]) for s in seeds]
            games = [b for _, b in pairs]
            counters = " ".join(f"{mean_of(games, k):>{w}.3f}" for k, w in
                                (("restarts_per_sim", 9), ("root_per_sim", 8), ("leaks_per_sim", 7)))
            sims = mean_of(games, "sims_per_move")
            deaths = sum(g["died"] for g in games)
            if name == control:
                print(f"{name:<20} {len(seeds):>5} {'(control)':>22} {'':>22} {deaths:>11} {sims:>10.0f} {counters}")
                continue

            def paired_ratio(statistic):
                def r(p):
                    base = statistic([x for x, _ in p])
                    return statistic([y for _, y in p]) / base if base else math.nan
                low, high = interval([r(rng.choices(pairs, k=len(pairs))) for _ in range(REPS)])
                return f"x{r(pairs):.3f} [{low:.3f}, {high:.3f}]"

            app_mean = lambda games: statistics.fmean(app(g) for g in games)
            control_deaths = sum(a["died"] for a, _ in pairs)
            print(f"{name:<20} {len(seeds):>5} {paired_ratio(app_mean):>22} {paired_ratio(per_piece):>22}"
                  f" {f'{deaths} vs {control_deaths}':>11} {sims:>10.0f} {counters}")


def line_fit(points):
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    slope = sum((x - mx) * (y - my) for x, y in points) / sxx
    return my - slope * mx, slope


def fit_report(configs, ladder_names, parallel, rng):
    seeds = sorted(set.intersection(*(set(configs[n]) for n in ladder_names + parallel)))
    sims = lambda g: g["sims_per_move"]

    def estimate(sample):
        mean = lambda n, f: statistics.fmean(f(configs[n][s]) for s in sample)
        x = [math.log2(parse(n)[1]) for n in ladder_names]
        a, b = line_fit(list(zip(x, [math.log(mean(n, app)) for n in ladder_names])))
        lad_sims = [(parse(n)[1], mean(n, sims)) for n in ladder_names]
        out = {}
        for name in parallel:
            eff = (math.log(mean(name, app)) - a) / b if b > 0 else math.nan
            sim = crossing(lad_sims, mean(name, sims))
            out[name] = (eff, sim, eff - sim if math.isfinite(sim) else math.nan)
        return b, out

    slope, point = estimate(seeds)
    boot = [estimate(rng.choices(seeds, k=len(seeds))) for _ in range(REPS)]
    slope_ci = interval([b for b, _ in boot])
    print(f"\nfit over the ladder: APP@H x{math.exp(slope):.3f} per doubling of think time"
          f" [{math.exp(slope_ci[0]):.3f}, {math.exp(slope_ci[1]):.3f}]; each rung's distance from the line:")
    a, b = line_fit([(math.log2(parse(n)[1]), math.log(statistics.fmean(app(configs[n][s]) for s in seeds)))
                     for n in ladder_names])
    print("  " + "  ".join(f"{parse(n)[1]}x {100 * (statistics.fmean(app(configs[n][s]) for s in seeds) / math.exp(a + b * math.log2(parse(n)[1])) - 1):+.1f}%"
                           for n in ladder_names))
    print(f"parallel configs from the fit ({len(seeds)} seeds)")
    print(f"{'config':<12} {'effective':>22} {'sims multiple':>24} {'strength efficiency':>22}")
    for name in parallel:
        cis = [interval([o[name][i] for _, o in boot]) for i in range(3)]
        mult = lambda v: "n/a" if not math.isfinite(v) else f"{2 ** v:.2f}x"
        eff, sim, ratio = point[name]
        print(f"{name:<12} {f'{mult(eff)} [{mult(cis[0][0])}, {mult(cis[0][1])}]':>22}"
              f" {f'{mult(sim)} [{mult(cis[1][0])}, {mult(cis[1][1])}]':>24}"
              f" {f'{show_ratio(ratio)} [{show_ratio(cis[2][0])}, {show_ratio(cis[2][1])}]':>22}")


def plot(out, dark, base_ms, horizon, names, ladder_names, parallel, stats, results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    if dark:
        surface, ink, muted, grid = "#1a1a19", "#ffffff", "#c3c2b7", "#2c2c2a"
        colours = {"ladder": "#3987e5", "parallel": "#d95926", "lf1": "#199e70"}
    else:
        surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
        colours = {"ladder": "#2a78d6", "parallel": "#eb6834", "lf1": "#1baf7a"}
    offsets = {"ladder": 0, "parallel": 0.12, "lf1": -0.12}

    def role(name):
        threads, _, lf = parse(name)
        return "parallel" if threads > 1 else ("lf1" if lf != 6 else "ladder")

    def label(name):
        threads, _, lf = parse(name)
        if threads > 1:
            return f"{threads} threads"
        return f"1 thread, LOAD_FACTOR {lf}" if lf != 6 else "1 thread"

    def x_of(name):
        return base_ms * parse(name)[1] * 2 ** offsets[role(name)]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5.2), width_ratios=[2.2, 1], facecolor=surface)
    for a in (ax, bx):
        a.set_facecolor(surface)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(muted)
        a.tick_params(colors=muted, labelcolor=muted)
        a.grid(True, color=grid, linewidth=0.8)
        a.set_axisbelow(True)

    jitter = random.Random(2)
    for name in names:
        xs = [x_of(name) * 2 ** jitter.uniform(-0.05, 0.05) for _ in stats[name]["values"]]
        ax.scatter(xs, stats[name]["values"], s=9, color=colours[role(name)], alpha=0.25, linewidths=0)

    ax.plot([x_of(n) for n in ladder_names], [stats[n]["mean"] for n in ladder_names],
            color=colours["ladder"], linewidth=2, zorder=3)
    shown = set()
    for name in names:
        s = stats[name]
        ax.errorbar([x_of(name)], [s["mean"]], yerr=[[s["mean"] - s["low"]], [s["high"] - s["mean"]]],
                    fmt="o", markersize=7, color=colours[role(name)], ecolor=colours[role(name)], elinewidth=2,
                    capsize=0, markeredgecolor=surface, markeredgewidth=2, zorder=4,
                    label=None if role(name) in shown else label(name))
        shown.add(role(name))

    for name in parallel:
        eff = results.get(name, (math.nan,))[0]
        if math.isfinite(eff):
            x_eff, y = base_ms * 2 ** eff, stats[name]["mean"]
            ax.plot([x_of(name), x_eff], [y, y], color=colours["parallel"], linewidth=1,
                    linestyle=(0, (4, 3)), zorder=2)
            ax.annotate(f"{label(name)} ≈ 1 thread at {2 ** eff:.1f}x", (x_eff, y),
                        xytext=(6, 8), textcoords="offset points", color=ink, fontsize=9)

    ax.set_xscale("log", base=2)
    ticks = sorted({base_ms * parse(n)[1] for n in names})
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(ticks[0] / 2 ** 0.5, ticks[-1] * 2 ** 0.5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("think time per move (ms)", color=muted)
    ax.set_ylabel(f"attack per piece over {horizon} pieces", color=muted)
    ax.set_title("APP@H: mean with 95% CI, dots are single games", color=ink, loc="left", fontsize=11)
    ax.legend(frameon=False, loc="lower right", labelcolor=ink)

    y = list(range(len(names)))
    widths = [pct(50 * (stats[n]["high"] - stats[n]["low"]), stats[n]["mean"]) for n in names]
    widths = [0 if math.isnan(w) else w for w in widths]
    bx.barh(y, widths, height=0.6, color=[colours[role(n)] for n in names])
    bx.set_yticks(y, [f"{label(n)}, {base_ms * parse(n)[1]} ms" for n in names], color=ink, fontsize=9)
    bx.invert_yaxis()
    bx.grid(axis="y", visible=False)
    for i, w in enumerate(widths):
        bx.annotate(f"±{w:.1f}%", (w, i), xytext=(4, 0), textcoords="offset points", va="center",
                    color=muted, fontsize=9)
    bx.set_xlabel("95% CI half-width, % of mean", color=muted)
    bx.set_title("Precision per config", color=ink, loc="left", fontsize=11)
    bx.set_xlim(0, max(max(widths) * 1.3, 1))

    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=surface)


if __name__ == "__main__":
    main()
