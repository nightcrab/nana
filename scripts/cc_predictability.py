"""How predictable are Cold Clear's attacks, and when they come, over the next few pieces?

Takes Cold Clear's positions from stadium games and plays Cold Clear forward from each one
(cc_replay) under many random continuations of the 7-bag, with a few repeats per continuation.
Repeats on one continuation differ only by the search's own randomness, so the spread splits into
what the unseen pieces cause and what the search causes. Every position also replays the pieces
the game really dealt, to check the replay against what Cold Clear did in the game. No garbage
arrives during a rollout, and Cold Clear's search ignores the meter, so a rollout is Cold Clear's
own plan for the position.

Attack per piece is binned as MarkovOpponent's emissions are: min(attack, 7), plus a death bin.

usage: python scripts/cc_predictability.py STADIUM.db [...] [--player 2] [--stride 5] [--max-positions 40]
           [--horizon 8] [--thinks 4000] [--queue-seeds 16] [--repeats 4] [--jobs 2] [--out FILE.jsonl]
       python scripts/cc_predictability.py --load FILE.jsonl     report on saved rollouts

cc_replay: cargo build --release -p cc-tbp --bin cc_replay --features tetrio_garbage (in ../cold-clear)
"""

import argparse
import json
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXE = ".exe" if os.name == "nt" else ""
DEFAULT_CC = ROOT.parent / "cold-clear" / "target" / "release" / f"cc_replay{EXE}"
PIECES = "IOTLJSZ"
ATTACKS = 8
DEATH = ATTACKS
BINS = ATTACKS + 1


def deal_sequence(rows):
    """every piece a player was dealt, in order, and how many were dealt by each row: each row's
    queue is [current, preview...], and the first hold takes two pieces off it"""
    first = rows[0]
    dealt = [first["current"]] + first["preview"]
    by_row = [len(dealt)]
    for prev, row in zip(rows, rows[1:]):
        old = [prev["current"]] + prev["preview"]
        new = [row["current"]] + row["preview"]
        shifts = [2, 1] if prev["hold"] is None and row["hold"] is not None else [1, 2]
        for s in shifts:
            if old[s:] == new[:len(new) - s]:
                break
        else:
            return None
        dealt += new[len(new) - s:]
        by_row.append(len(dealt))
    for start in range(0, len(dealt) - len(dealt) % 7, 7):
        if sorted(dealt[start:start + 7]) != sorted(PIECES):
            return None
    return dealt, by_row


def load_positions(paths, player, stride, horizon, limit):
    p = f"p{player}"
    cols = ["game_id", "move_index", "state", f"{p}_board", f"{p}_current_piece", f"{p}_hold", f"{p}_combo",
            f"{p}_b2b", f"{p}_attack", f"{p}_meter"] + [f"{p}_queue_{i}" for i in range(5)]
    positions = []
    skipped = 0
    for path in paths:
        db = sqlite3.connect(path)
        games = defaultdict(list)
        for r in db.execute(f"SELECT {', '.join(cols)} FROM Data ORDER BY game_id, move_index"):
            if r[2] != "PLAYING":
                continue
            games[r[0]].append({
                "move": r[1], "board": r[3], "current": r[4], "hold": None if r[5] == "NULL" else r[5],
                "combo": r[6], "b2b": r[7], "attack": r[8], "meter": r[9], "preview": list(r[10:15]),
            })
        for game_id, rows in games.items():
            seq = deal_sequence(rows)
            if seq is None:
                skipped += 1
                continue
            dealt, by_row = seq
            for k in range(0, len(rows), stride):
                row = rows[k]
                n = by_row[k]
                cells = row["board"]
                assert len(cells) == 200, "expected a 10x20 board"
                board = [[1 if cells[x + y * 10] else 0 for x in range(10)] for y in range(20)]
                board += [[0] * 10 for _ in range(20)]
                bag = [x for x in PIECES if x not in dealt[n - n % 7:n]] if n % 7 else []
                positions.append({
                    "source": f"{Path(path).name}:{game_id}:{row['move']}",
                    "board": board, "queue": [row["current"]] + row["preview"], "hold": row["hold"],
                    "bag": bag, "b2b": row["b2b"], "combo": row["combo"], "meter": row["meter"],
                    "future": dealt[n:n + 2 * horizon],
                    "recorded": [r["attack"] for r in rows[k:k + horizon]],
                    "height": max([y + 1 for y in range(20) for x in range(10) if board[y][x]], default=0),
                })
    if skipped:
        print(f"skipped {skipped} games whose queues didn't read as a 7-bag", file=sys.stderr)
    if limit and len(positions) > limit:
        step = len(positions) / limit
        positions = [positions[int(i * step)] for i in range(limit)]
    return positions


def run_replays(positions, args):
    exe = Path(args.cc)
    if not exe.exists():
        sys.exit(f"{exe} not found; build it in ../cold-clear with\n"
                 "  cargo build --release -p cc-tbp --bin cc_replay --features tetrio_garbage")
    with tempfile.TemporaryDirectory() as tmp:
        file = Path(tmp) / "positions.json"
        file.write_text(json.dumps({"positions": positions}))
        cmd = [str(exe), "--positions", str(file), "--horizon", str(args.horizon), "--thinks", str(args.thinks),
               "--queue-seeds", str(args.queue_seeds), "--repeats", str(args.repeats), "--seed", str(args.seed),
               "--jobs", str(args.jobs)]
        total = len(positions) * (args.queue_seeds + 1) * args.repeats
        rollouts = []
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
        for line in proc.stdout:
            rollouts.append(json.loads(line))
            if len(rollouts) % 50 == 0 or len(rollouts) == total:
                print(f"\r{len(rollouts)}/{total} rollouts", end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
        if proc.wait() != 0:
            sys.exit(f"cc_replay failed ({proc.returncode})")
    return rollouts


def binned(moves, horizon):
    out = [min(m["attack"], ATTACKS - 1) for m in moves[:horizon]]
    return out + [DEATH] * (horizon - len(out))


def entropy(counts, correct=True):
    """bits; Miller-Madow corrected unless correct is False"""
    n = sum(counts.values())
    h = -sum(c / n * math.log2(c / n) for c in counts.values() if c)
    if correct:
        h += (sum(1 for c in counts.values() if c) - 1) / (2 * n * math.log(2))
    return h


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def variance_split(groups):
    """law of total variance over groups of equal size: (total, within-group share)"""
    values = [v for g in groups for v in g]
    m = mean(values)
    total = mean((v - m) ** 2 for v in values)
    within = mean(mean((v - mean(g)) ** 2 for v in g) * len(g) / max(len(g) - 1, 1) for g in groups)
    return total, within


def report(positions, rollouts, horizon):
    by_pos = defaultdict(lambda: defaultdict(list))
    for r in rollouts:
        by_pos[r["position"]][r["queue_seed"]].append(r["moves"])
    H = horizon

    print(f"{len(positions)} positions, horizon {H}, attack bins 0..{ATTACKS - 1} + death\n")
    print(f"{'position':<26}{'h':>3}{'b2b':>4}  {'game':>5} {'true q':>7}  {'random queue: total attack':>27}"
          f"  {'P(any)':>6}  {'first attack (piece: share)':<30}")
    pooled = Counter()
    cond_h = defaultdict(list)
    cond_h_search = defaultdict(list)
    p_attack = defaultdict(list)
    timing_acc = defaultdict(list)
    modal_seq = []
    total_var = []
    within_var = []
    hit_var = []
    hit_within = []
    agree = []
    agree_total = []
    for p, pos in enumerate(positions):
        seeds = by_pos[p]
        rand = [m for q, ms in seeds.items() if q >= 0 for m in ms]
        if not rand:
            continue
        seqs = [binned(m, H) for m in rand]
        totals = [sum(x["attack"] for x in m[:H]) for m in rand]
        for s in seqs:
            pooled.update(s)
        for k in range(H):
            col = Counter(s[k] for s in seqs)
            cond_h[k].append(entropy(col))
            hit = mean(1.0 if s[k] not in (0, DEATH) else 0.0 for s in seqs)
            p_attack[k].append(hit)
            timing_acc[k].append(max(hit, 1 - hit))
            for q, ms in seeds.items():
                if q >= 0 and len(ms) > 1:
                    cond_h_search[k].append(entropy(Counter(binned(m, H)[k] for m in ms), correct=False))
        modal_seq.append(Counter(map(tuple, seqs)).most_common(1)[0][1] / len(seqs))
        groups = [[sum(x["attack"] for x in m[:H]) for m in ms] for q, ms in seeds.items() if q >= 0]
        t, w = variance_split(groups)
        total_var.append(t)
        within_var.append(w)
        hits = [[sum(1 for x in binned(m, H) if x not in (0, DEATH)) for m in ms] for q, ms in seeds.items() if q >= 0]
        t, w = variance_split(hits)
        hit_var.append(t)
        hit_within.append(w)

        firsts = Counter(next((k + 1 for k, x in enumerate(s) if x not in (0, DEATH)), None) for s in seqs)
        first_txt = ", ".join(f"{'-' if k is None else k}: {c / len(seqs):.0%}" for k, c in firsts.most_common(3))
        recorded = pos.get("recorded", [])
        true = seeds.get(-1, [])
        true_total = mean(sum(x["attack"] for x in m[:H]) for m in true) if true else float("nan")
        if true and len(recorded) == H:
            rec_hits = [a > 0 for a in recorded]
            for m in true:
                got = [x not in (0, DEATH) for x in binned(m, H)]
                agree.append(mean(1.0 if a == b else 0.0 for a, b in zip(got, rec_hits)))
                agree_total.append(sum(x["attack"] for x in m[:H]) == sum(recorded))
        sd = math.sqrt(mean((x - mean(totals)) ** 2 for x in totals))
        game_total = sum(recorded) if len(recorded) == H else float("nan")
        print(f"{pos['source']:<26}{pos['height']:>3}{pos['b2b']:>4}  {game_total:>5.0f} {true_total:>7.1f}"
              f"  {mean(totals):>13.1f} ± {sd:<4.1f} [{min(totals)}-{max(totals)}]"
              f"  {mean(1.0 if t else 0.0 for t in totals):>6.0%}  {first_txt:<30}")

    h0 = entropy(pooled, correct=False)
    print(f"\nper-piece attack entropy, bits (lower = more predictable)")
    print(f"  no knowledge of the position (pooled marginal): {h0:.3f}")
    print(f"  {'piece':>5}  {'H(a_k | position)':>18}  {'H(a_k | position, queue)':>25}  {'P(attack)':>9}"
          f"  {'best yes/no guess':>17}")
    for k in range(H):
        print(f"  {k + 1:>5}  {mean(cond_h[k]):>18.3f}  {mean(cond_h_search[k]):>25.3f}  {mean(p_attack[k]):>9.2f}"
              f"  {mean(timing_acc[k]):>17.0%}")
    for n in sorted({min(5, H), H}):
        c = mean(mean(cond_h[k]) for k in range(n))
        s = mean(mean(cond_h_search[k]) for k in range(n))
        print(f"  first {n}: {c:.3f} bits/piece given the position ({1 - c / h0:.0%} below no knowledge),"
              f" {s:.3f} given the queue too")
    print("  (the per-piece marginals bound the joint sequence's entropy per piece from above; the queue column"
          " is a plug-in over the repeats, so it reads low)")

    tv, wv = mean(total_var), mean(within_var)
    hv, hw = mean(hit_var), mean(hit_within)
    print(f"\nspread within a position, averaged over positions")
    print(f"  {H}-piece total attack: variance {tv:.2f} (sd {math.sqrt(tv):.2f});"
          f" search noise alone {wv:.2f} ({wv / tv if tv else 0:.0%}), unseen pieces the rest")
    print(f"  pieces that attack:   variance {hv:.2f} (sd {math.sqrt(hv):.2f});"
          f" search noise alone {hw:.2f} ({hw / hv if hv else 0:.0%})")
    print(f"  the most common exact {H}-piece attack pattern covers {mean(modal_seq):.0%} of rollouts")
    if agree:
        print(f"\nreplay on the game's own pieces vs the game: attack-or-not agrees on {mean(agree):.0%} of pieces,"
              f" {H}-piece total matches exactly in {mean(agree_total):.0%} of replays"
              f" (the game had garbage arriving and its own damage table)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dbs", nargs="*")
    ap.add_argument("--player", type=int, default=2, help="which stadium side is Cold Clear")
    ap.add_argument("--stride", type=int, default=5, help="take every Nth move of each game")
    ap.add_argument("--max-positions", type=int, default=40)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--thinks", type=int, default=4000, help="Cold Clear thinks per piece")
    ap.add_argument("--queue-seeds", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=4, help="rollouts per queue")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--cc", default=str(DEFAULT_CC))
    ap.add_argument("--out", help="save positions and rollouts here (jsonl)")
    ap.add_argument("--load", help="report on a saved run instead")
    args = ap.parse_args()

    if args.load:
        lines = [json.loads(l) for l in open(args.load)]
        head = lines[0]
        report(head["positions"], lines[1:], head["horizon"])
        return
    if not args.dbs:
        ap.error("give stadium databases or --load")
    positions = load_positions(args.dbs, args.player, args.stride, args.horizon, args.max_positions)
    if not positions:
        sys.exit("no positions")
    print(f"{len(positions)} positions; {len(positions) * (args.queue_seeds + 1) * args.repeats} rollouts of"
          f" {args.horizon} pieces at {args.thinks} thinks", file=sys.stderr)
    rollouts = run_replays(positions, args)
    if args.out:
        with open(args.out, "w") as f:
            f.write(json.dumps({"positions": positions, "horizon": args.horizon, "thinks": args.thinks}) + "\n")
            for r in rollouts:
                f.write(json.dumps(r) + "\n")
    report(positions, rollouts, args.horizon)


if __name__ == "__main__":
    main()
