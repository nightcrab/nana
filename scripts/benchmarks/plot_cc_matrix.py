#!/usr/bin/env python3
"""Plot a cc_matrix.py run as a heatmap for the README, light and dark.

    plot_cc_matrix.py OUT [--out docs]
"""
import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", secondary="#52514e", muted="#898781", low="#f1f0ec",
                  high="#2a78d6", on_high="#ffffff", flip=0.75),
    "dark": dict(surface="#1a1a19", ink="#ffffff", secondary="#c3c2b7", muted="#898781", low="#262624",
                 high="#3987e5", on_high="#ffffff", flip=0.55),
}
FLOOR, CEIL = 0.50, 0.66  # colour scale: even score to the best cell seen


def font():
    for name in ("Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "sans-serif"


def wilson(p, g, z=1.96):
    den = 1 + z * z / g
    mid = (p + z * z / (2 * g)) / den
    half = z * ((p * (1 - p) / g + z * z / (4 * g * g)) ** 0.5) / den
    return mid - half, mid + half


def load(out):
    threads = json.load(open(out / "matrix.json"))["threads"]
    meta = json.load(open(out / "matrix.json"))
    cells = json.load(open(out / "cells.json"))
    grid = {}
    for n in threads:
        for c in threads:
            nana, cc, dr = cells[f"n{n}_c{c}"]
            g = nana + cc + dr
            p = (nana + dr / 2) / g
            grid[n, c] = (p, *wilson(p, g), g)
    return threads, meta, grid


def plot(path, t, f, threads, meta, grid):
    k = len(threads)
    fig = plt.figure(figsize=(2048 / 150, 1300 / 150), dpi=150, facecolor=t["surface"])
    ax = fig.add_axes([0.12, 0.06, 0.76, 0.70])
    ax.set_facecolor(t["surface"])
    cmap = LinearSegmentedColormap.from_list("score", [t["low"], t["high"]])
    games = min(v[3] for v in grid.values())
    for i, n in enumerate(threads):
        for j, c in enumerate(threads):
            p, lo, hi, _ = grid[n, c]
            w = min(max((p - FLOOR) / (CEIL - FLOOR), 0), 1)
            ax.add_patch(plt.Rectangle((j + 0.03, i + 0.03), 0.94, 0.94, color=cmap(w), linewidth=0))
            ink = t["on_high"] if w > t["flip"] else t["ink"]
            sub = t["on_high"] if w > t["flip"] else t["secondary"]
            ax.text(j + 0.5, i + 0.44, f"{100 * p:.1f}%", ha="center", va="center", fontsize=17,
                    fontweight="semibold", color=ink, fontfamily=f)
            ax.text(j + 0.5, i + 0.72, f"{100 * lo:.0f}–{100 * hi:.0f}", ha="center", va="center", fontsize=11,
                    color=sub, fontfamily=f)
    ax.set_xlim(0, k)
    ax.set_ylim(k, 0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks([j + 0.5 for j in range(k)], [str(c) for c in threads])
    ax.set_yticks([i + 0.5 for i in range(k)], [str(n) for n in threads])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(colors=t["muted"], labelcolor=t["muted"], labelsize=13, length=0, pad=8)
    ax.set_xlabel("Cold Clear threads", color=t["muted"], fontsize=13, fontfamily=f, labelpad=10)
    ax.set_ylabel("nana threads", color=t["muted"], fontsize=13, fontfamily=f, labelpad=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(f)
    ms = 1000 / meta["pps"]
    fig.text(0.031, 0.955, "nana vs Cold Clear by thread count", fontsize=20, fontweight="semibold",
             color=t["ink"], fontfamily=f, va="top")
    fig.text(0.031, 0.905, f"nana's score (wins + half of draws), {games} games per cell, {ms:g} ms per piece "
             f"on Universal Shaktris Stadium · 95% interval below each score", fontsize=13, color=t["secondary"],
             fontfamily=f, va="top")
    fig.savefig(path, dpi=150, facecolor=t["surface"])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--out", dest="docs", type=pathlib.Path, default=pathlib.Path("docs"))
    a = ap.parse_args()
    threads, meta, grid = load(a.out)
    f = font()
    for name, t in THEMES.items():
        plot(a.docs / f"cc-matrix-{name}.png", t, f, threads, meta, grid)


if __name__ == "__main__":
    main()
