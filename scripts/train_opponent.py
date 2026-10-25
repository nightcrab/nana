"""One-command cached datagen, CPU CNN training, and untouched-game evaluation.

See scripts/opponent-training.md. Run from any directory with Python and CPU PyTorch.
"""

import argparse
import copy
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from opponent_data import generate, digest_file
from opponent_model import Predictor, BINS

ROOT = Path(__file__).resolve().parent.parent


def split_games(games, seed):
    ids = np.array(sorted(set(games)))
    if len(ids) < 3:
        raise ValueError("need at least three distinct games for train/validation/test")
    np.random.default_rng(seed).shuffle(ids)
    held = max(1, int(len(ids)*0.15))
    return {name: np.flatnonzero(np.isin(games, selected)) for name, selected in
            (("train", ids[2*held:]), ("validation", ids[:held]), ("test", ids[held:2*held]))}


def sequence_loss(logits, real, real_mask, replay, replay_mask, weight):
    logp = logits.log_softmax(-1)
    real_nll = -logp.gather(-1, real[..., None]).squeeze(-1)
    loss = (real_nll * real_mask).sum() / real_mask.sum().clamp_min(1)
    if weight > 0 and replay.shape[1]:
        # Average samples within each position: repeats do not increase a board's weight.
        lp = logp[:, None].expand(-1, replay.shape[1], -1, -1)
        nll = -lp.gather(-1, replay[..., None]).squeeze(-1).mean(1)
        loss = loss + weight * (nll * replay_mask).sum() / replay_mask.sum().clamp_min(1)
    return loss


def predict(model, data, indices, batch):
    model.eval()
    outputs = []
    with torch.no_grad():
        for selected in indices.split(batch):
            outputs.append(model(data["board"][selected], data["context"][selected]).softmax(-1))
    return torch.cat(outputs)


def metrics(prob, target, mask):
    p = prob.gather(-1, target[..., None]).squeeze(-1).clamp_min(1e-12)
    bits = -p.log2()
    attack_prob = prob[..., 1:8].sum(-1)
    attack = ((target > 0) & (target < 8)).float()
    values = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0], dtype=prob.dtype)
    bias = (prob * values).sum(-1) - values[target]
    result = {}
    for horizon in sorted({1, min(5, target.shape[1]), target.shape[1]}):
        valid = mask[:, :horizon]
        denom = valid.sum().clamp_min(1)
        result[str(horizon)] = dict(
            bits_per_piece=float((bits[:, :horizon]*valid).sum()/denom),
            capped_attack_bias=float((bias[:, :horizon]*valid).sum()/denom),
            attack_brier=float((((attack_prob-attack)**2)[:, :horizon]*valid).sum()/denom),
            observations=int(denom),
        )
    return result


def gain_intervals(cnn, baseline, target, mask, games, seed):
    """Bootstrap whole games, keeping correlated/overlapping windows together."""
    ids = sorted(set(games))
    if len(ids) < 2:
        return {str(h): None for h in sorted({1, min(5, target.shape[1]), target.shape[1]})}
    gains = (cnn.gather(-1, target[..., None]).squeeze(-1).clamp_min(1e-12).log2()
             - baseline.gather(-1, target[..., None]).squeeze(-1).clamp_min(1e-12).log2()).numpy()
    valid = mask.numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(ids), size=(1000, len(ids)))
    result = {}
    for h in sorted({1, min(5, target.shape[1]), target.shape[1]}):
        sums = np.array([(gains[games == gid, :h]*valid[games == gid, :h]).sum() for gid in ids])
        counts = np.array([valid[games == gid, :h].sum() for gid in ids])
        samples = sums[draws].sum(1)/counts[draws].sum(1).clip(min=1)
        result[str(h)] = np.quantile(samples, [0.025, 0.975]).tolist()
    return result


def fit(args, data, splits, use_board):
    label = "cnn" if use_board else "context"
    torch.manual_seed(args.seed)
    model = Predictor(args.horizon, args.width, args.blocks, data["context"].shape[-1], board=use_board)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = torch.Generator().manual_seed(args.seed)
    best_loss, best, best_epoch, stale = math.inf, None, 0, 0
    history = []
    print(f"{label}: {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    for epoch in range(1, args.epochs+1):
        start = time.perf_counter()
        model.train()
        indices = splits["train"][torch.randperm(len(splits["train"]), generator=rng)]
        loss_sum = 0.0
        for selected in indices.split(args.batch_size):
            logits = model(data["board"][selected], data["context"][selected])
            loss = sequence_loss(logits, data["real"][selected], data["real_mask"][selected],
                                 data["replay"][selected], data["replay_mask"][selected], args.replay_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(selected)
        val = splits["validation"]
        prob = predict(model, data, val, args.batch_size)
        val_bits = metrics(prob, data["real"][val], data["real_mask"][val])[str(args.horizon)]["bits_per_piece"]
        elapsed = time.perf_counter()-start
        row = dict(epoch=epoch, loss=loss_sum/len(indices), validation_bits=val_bits, seconds=elapsed)
        history.append(row)
        print(f"{label} epoch {epoch}: loss {row['loss']:.4f}, val {val_bits:.4f} bits/piece, {elapsed:.1f}s", flush=True)
        if val_bits < best_loss:
            best_loss, best, best_epoch, stale = val_bits, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best)
    torch.save(dict(state_dict=best, epoch=best_epoch, args=vars(args),
                    context_size=data["context"].shape[-1], board=use_board), Path(args.out)/f"{label}.pt")
    return model, history


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dbs", nargs="+", help="stadium SQLite databases (target side must be known)")
    p.add_argument("--out", default="results/opponent-cnn")
    p.add_argument("--cache", default="results/opponent-data")
    p.add_argument("--player", choices=("1", "2", "auto"), default="2",
                   help="target side; auto reads Cold Clear's side from adjacent sides.txt")
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--max-positions", type=int, default=0, help="per database; 0 uses all")
    p.add_argument("--queue-seeds", type=int, default=8)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--thinks", type=int, default=1000)
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2)//2))
    p.add_argument("--shard-size", type=int, default=32)
    p.add_argument("--cc", default=None, help="opponent-replay binary; builds the local helper if omitted")
    p.add_argument("--replay-weight", type=float, default=0.5, help="0 skips replay generation")
    p.add_argument("--pieces", action="store_true", help="also observe hold/current/five preview pieces")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1)
    return p


def main():
    args = parser().parse_args()
    for name in ("horizon", "stride", "queue_seeds", "repeats", "thinks", "jobs", "shard_size",
                 "width", "epochs", "patience", "batch_size", "threads"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.replay_weight < 0 or args.blocks < 0 or args.max_positions < 0:
        raise ValueError("replay weight, blocks, and position limit must be nonnegative")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.replay_weight and args.cc is None:
        manifest = ROOT/"scripts/opponent_replay/Cargo.toml"
        subprocess.run(["cargo", "build", "--release", "--manifest-path", str(manifest)], check=True)
        args.cc = str(manifest.parent/"target/release"/("opponent-replay.exe" if os.name == "nt" else "opponent-replay"))
    torch.set_num_threads(args.threads)
    arrays, manifest = generate(args)
    splits_np = split_games(arrays["game"], args.seed)
    splits = {k: torch.from_numpy(v) for k, v in splits_np.items()}
    print("positions by split: " + str({k: len(v) for k, v in splits.items()}), flush=True)
    data = {k: torch.from_numpy(v) for k, v in arrays.items() if k != "game"}
    # Both baselines use training games only. No validation/test replay samples enter fitting.
    train = splits["train"]
    counts = torch.zeros(args.horizon, BINS) + 1
    for h in range(args.horizon):
        valid = data["real_mask"][train, h]
        counts[h] += torch.bincount(data["real"][train, h][valid], minlength=BINS)
    pooled = counts/counts.sum(-1, keepdim=True)
    models, histories = {}, {}
    for use_board in (False, True):
        label = "cnn" if use_board else "context"
        models[label], histories[label] = fit(args, data, splits, use_board)
    test = splits["test"]
    probabilities = {label: predict(model, data, test, args.batch_size) for label, model in models.items()}
    probabilities["pooled"] = pooled[None].expand(len(test), -1, -1)
    report = {label: metrics(p, data["real"][test], data["real_mask"][test])
              for label, p in probabilities.items()}
    for horizon, row in report["cnn"].items():
        row["gain_vs_pooled"] = report["pooled"][horizon]["bits_per_piece"]-row["bits_per_piece"]
        row["gain_vs_context"] = report["context"][horizon]["bits_per_piece"]-row["bits_per_piece"]
    test_games = arrays["game"][splits_np["test"]]
    for baseline in ("pooled", "context"):
        intervals = gain_intervals(probabilities["cnn"], probabilities[baseline], data["real"][test],
                                   data["real_mask"][test], test_games, args.seed)
        for horizon, interval in intervals.items():
            report["cnn"][horizon][f"gain_vs_{baseline}_95pct_game_bootstrap"] = interval
    result = dict(args=vars(args), data=manifest,
                  split_games={k: sorted(set(arrays["game"][v])) for k, v in splits_np.items()},
                  history=histories, test=report,
                  source_hashes={f: digest_file(Path(__file__).parent/f) for f in
                                 ("train_opponent.py", "opponent_data.py", "opponent_model.py")})
    (Path(args.out)/"report.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(Path(args.out)/"test-predictions.npz", **{k: v.numpy() for k, v in probabilities.items()},
                        targets=data["real"][test].numpy(), mask=data["real_mask"][test].numpy(),
                        games=arrays["game"][splits_np["test"]])
    print("Untouched stadium test games:", flush=True)
    print("pieces   CNN bits/piece   gain vs pooled   gain vs context", flush=True)
    for horizon, row in report["cnn"].items():
        print(f"{horizon:>6}   {row['bits_per_piece']:>14.4f}   {row['gain_vs_pooled']:>14.4f}"
              f"   {row['gain_vs_context']:>15.4f}", flush=True)
    print(f"Saved checkpoints, predictions and report to {args.out}", flush=True)


if __name__ == "__main__":
    main()
