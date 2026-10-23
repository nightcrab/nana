#!/usr/bin/env python3
"""Merge strength_ladder.sh task outputs into one file per config.

    strength_merge.py DIR BASE_MS

Reads DIR/parts/<config>-seed<seed>.jsonl, where config is t<threads>-x<multiple>-lf<load
factor>[-v<variant>], and writes DIR/<config>.jsonl sorted by seed, with config, mult, base_ms,
load_factor and variant ("" for the plain build) added to every game.
"""

import collections
import glob
import json
import os
import re
import sys


def main():
    out, base_ms = sys.argv[1], int(sys.argv[2])
    configs = collections.defaultdict(list)
    for path in glob.glob(os.path.join(out, "parts", "*.jsonl")):
        name = re.sub(r"-seed\d+\.jsonl$", "", os.path.basename(path))
        match = re.fullmatch(r"t(\d+)-x(\d+)-lf(\d+)(?:-v([\w.-]+))?", name)
        mult, lf, variant = int(match[2]), int(match[3]), match[4] or ""
        # utf-8-sig: PowerShell's Out-File may start the file with a byte order mark
        for line in open(path, encoding="utf-8-sig"):
            if line.startswith("{"):
                try:
                    game = json.loads(line)
                except json.JSONDecodeError as error:
                    # e.g. two processes wrote the file at once; drop the game rather than the run
                    print(f"skipping a corrupt line in {path}: {error}", file=sys.stderr)
                    continue
                game.update(config=name, mult=mult, base_ms=base_ms, load_factor=lf, variant=variant)
                configs[name].append(game)
    for name, rows in configs.items():
        seeds = collections.Counter(game["seed"] for game in rows)
        twice = sorted(seed for seed, count in seeds.items() if count > 1)
        if twice:
            print(f"warning: {name} has seeds played more than once: {twice}", file=sys.stderr)
        with open(os.path.join(out, name + ".jsonl"), "w") as f:
            for game in sorted(rows, key=lambda g: g["seed"]):
                f.write(json.dumps(game) + "\n")


if __name__ == "__main__":
    main()
