"""Recorded stadium targets plus cached CC continuations under fixed recorded incoming attacks."""

import hashlib
import json
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path

import numpy as np

import cc_predictability as cc

VERSION = 1


def target_player(path, requested):
    if requested != "auto":
        return int(requested)
    sides_path = Path(path).parent / "sides.txt"
    sides = sides_path.read_text().split()
    if len(sides) != 4 or sides[0] != "p1" or sides[2] != "p2":
        raise ValueError(f"unexpected side metadata in {sides_path}")
    matches = [i for i, name in ((1, sides[1]), (2, sides[3]))
               if Path(name).name in ("cc-tbp", "cc-tbp.exe")]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one Cold Clear side in {sides_path}")
    return matches[0]


def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def positions_from_db(path, player, horizon, stride):
    """Only zero-delay stadium games are supported. Future inputs never enter the encoder."""
    p, other = f"p{player}", f"p{3-player}"
    with closing(sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        games = defaultdict(list)
        for row in db.execute("SELECT * FROM Data ORDER BY game_id, move_index"):
            games[row["game_id"]].append(row)
    positions = []
    skipped = 0
    for gid, all_rows in games.items():
        rows = [r for r in all_rows if r["state"] == "PLAYING"]
        terminal = [r for r in all_rows if r["state"] != "PLAYING"]
        if not rows:
            continue
        queue_rows = [{"current": r[f"{p}_current_piece"],
                       "preview": [r[f"{p}_queue_{i}"] for i in range(5)],
                       "hold": None if r[f"{p}_hold"] in (None, "NULL") else r[f"{p}_hold"]}
                      for r in rows]
        sequence = cc.deal_sequence(queue_rows)
        if sequence is None:
            skipped += 1
            continue
        dealt, by_row = sequence
        lost = bool(terminal and terminal[-1]["state"] in ("DRAW", f"P{3-player}_WIN"))
        for k in range(0, len(rows), stride):
            r = rows[k]
            board = np.frombuffer(r[f"{p}_board"], dtype=np.uint8).reshape(20, 10)
            q = queue_rows[k]
            n = by_row[k]
            future_rows = rows[k:k+horizon]
            incoming = [max(0, int(s[f"{other}_attack"]) - int(s[f"{other}_meter"]))
                        for s in future_rows]
            # Gross attacks, matching stadium's attack column and the entropy script.
            recorded = [min(int(s[f"{p}_attack"]), 7) for s in future_rows]
            remaining = len(recorded)
            real_mask = [True] * remaining + [lost] * (horizon-remaining)
            recorded += [8 if lost else 0] * (horizon-remaining)
            positions.append(dict(
                game=str(gid), move=int(r["move_index"]),
                board=(board != 0).astype(int).tolist() + [[0]*10 for _ in range(20)],
                queue=[q["current"]] + q["preview"], hold=q["hold"],
                bag=[x for x in cc.PIECES if x not in dealt[n-n%7:n]] if n%7 else [],
                b2b=int(r[f"{p}_b2b"]), combo=int(r[f"{p}_combo"]),
                meter=int(r[f"{p}_meter"]), incoming=incoming,
                recorded=recorded, real_mask=real_mask,
                replay_mask=[i < remaining for i in range(horizon)],
            ))
    print(f"{path}: {len(positions)} positions; {skipped} games skipped for invalid piece sequences", flush=True)
    return positions


def context(position, pieces=False):
    values = [np.log1p(position["combo"]), np.log1p(position["b2b"]), np.log1p(position["meter"])]
    if pieces:
        for piece in [position["hold"]] + position["queue"]:
            values.extend(float(piece == name) for name in cc.PIECES)
    return values


def generate(args):
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    arrays = defaultdict(list)
    manifest = []
    replay_signature = None
    if args.replay_weight > 0:
        replay_signature = digest_file(args.cc)
    for db_path in args.dbs:
        player = target_player(db_path, args.player)
        db_hash = digest_file(db_path)
        spec = dict(version=VERSION, db=db_hash, player=player, horizon=args.horizon,
                    stride=args.stride, limit=args.max_positions, seed=args.seed,
                    queue_seeds=args.queue_seeds, repeats=args.repeats, thinks=args.thinks,
                    replay=replay_signature, loader=digest_file(__file__),
                    entropy_loader=digest_file(cc.__file__))
        key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
        directory = cache / key
        directory.mkdir(exist_ok=True)
        position_file = directory / "positions.json"
        if position_file.exists():
            positions = json.loads(position_file.read_text())["positions"]
            print(f"cached positions: {directory}", flush=True)
        else:
            positions = positions_from_db(db_path, player, args.horizon, args.stride)
            if args.max_positions and len(positions) > args.max_positions:
                index = np.random.default_rng(args.seed).choice(len(positions), args.max_positions, replace=False)
                positions = [positions[i] for i in sorted(index)]
            temp = position_file.with_suffix(".tmp")
            temp.write_text(json.dumps(dict(spec=spec, positions=positions)))
            temp.replace(position_file)
        for start in range(0, len(positions), args.shard_size):
            shard = positions[start:start+args.shard_size]
            target_file = directory / f"replay-{start:07d}-{len(shard)}.npz"
            if args.replay_weight <= 0:
                targets = np.empty((len(shard), 0, args.horizon), dtype=np.int64)
            elif target_file.exists():
                with np.load(target_file) as saved:
                    targets = saved["targets"]
                print(f"cached replay: {target_file.name}", flush=True)
            else:
                print(f"replaying {start+1}..{start+len(shard)} / {len(positions)}", flush=True)
                replays = cc.run_replays(shard, args)
                targets = np.full((len(shard), args.queue_seeds*args.repeats, args.horizon), -1, dtype=np.int64)
                for replay in replays:
                    slot = replay["queue_seed"] * args.repeats + replay["repeat"]
                    if not (0 <= replay["position"] < len(shard) and 0 <= slot < targets.shape[1]):
                        raise ValueError("replay returned an invalid position or sample ID")
                    if (targets[replay["position"], slot] >= 0).any():
                        raise ValueError("duplicate replay")
                    targets[replay["position"], slot] = cc.binned(replay["moves"], args.horizon)
                if (targets < 0).any():
                    raise ValueError("incomplete replay shard")
                temp = target_file.with_suffix(".tmp")
                with open(temp, "wb") as f:
                    np.savez_compressed(f, targets=targets)
                temp.replace(target_file)
            arrays["replay"].extend(targets)
            for position in shard:
                arrays["board"].append(position["board"][:20])
                arrays["context"].append(context(position, args.pieces))
                arrays["real"].append(position["recorded"])
                arrays["real_mask"].append(position["real_mask"])
                arrays["replay_mask"].append(position["replay_mask"])
                arrays["game"].append(db_hash + ":" + position["game"])
        manifest.append(dict(path=str(Path(db_path).resolve()), key=key, player=player, positions=len(positions)))
    if not arrays["board"]:
        raise ValueError("no complete horizon windows found in the stadium databases")
    out = {k: np.asarray(v) for k, v in arrays.items()}
    out["board"] = out["board"].astype(np.float32)[:, None]
    out["context"] = out["context"].astype(np.float32)
    out["real"] = out["real"].astype(np.int64)
    out["replay"] = out["replay"].astype(np.int64)
    return out, manifest
