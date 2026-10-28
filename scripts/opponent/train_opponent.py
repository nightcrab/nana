"""One-command cached datagen, CPU CNN training, and untouched-game evaluation.

Run from any directory with Python and CPU PyTorch.
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

from opponent_data import generate, digest_file
from opponent_model import Predictor, BINS

ROOT = Path(__file__).resolve().parents[2]


def split_games(games, seed):
    ids = np.array(sorted(set(games)))
    if len(ids) < 3:
        raise ValueError("need at least three distinct games for train/validation/test")
    np.random.default_rng(seed).shuffle(ids)
    held = max(1, int(len(ids)*0.15))
    return {name: np.flatnonzero(np.isin(games, selected)) for name, selected in
            (("train", ids[2*held:]), ("validation", ids[:held]), ("test", ids[held:2*held]))}


def sequence_loss(logits, real, real_mask, replay, replay_mask, weight, replay_logits=None):
    logp = logits.log_softmax(-1)
    real_nll = -logp.gather(-1, real[..., None]).squeeze(-1)
    loss = real_nll.masked_fill(~real_mask, 0).sum() / real_mask.sum().clamp_min(1)
    if weight > 0 and replay.shape[1]:
        # Average samples within each position: repeats do not increase a board's weight.
        if replay_logits is None:
            raise ValueError("autoregressive replay loss requires logits from each replay's own prefix")
        lp = replay_logits.log_softmax(-1)
        nll = -lp.gather(-1, replay[..., None]).squeeze(-1).mean(1)
        loss = loss + weight * nll.masked_fill(~replay_mask, 0).sum() / replay_mask.sum().clamp_min(1)
    return loss


def moment_losses(marginals, targets):
    """Per-root APP and per-offset bucket frequencies from complete continuations."""
    frequencies = torch.nn.functional.one_hot(targets, BINS).to(marginals.dtype).mean(1)
    values = marginals.new_tensor([0, 1, 2, 3, 4, 5, 6, 7, 0])
    app_error = ((marginals-frequencies)*values).sum(-1).mean(-1)
    return app_error.square().mean(), (marginals-frequencies).square().sum(-1).mean()


def grouped_batches(indices, batch_size, samples):
    """Keep all cached continuations at a root together without changing the cache."""
    offsets = torch.arange(samples)
    for roots in indices.split(max(1, batch_size//samples)):
        yield (roots[:, None]*samples + offsets).flatten()


@torch.no_grad()
def validate_moments(model, data, indices, batch_size, samples):
    app_sum = bucket_sum = count = 0
    # Fixed draws make epoch comparisons reproducible without changing training RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(10001)
        for selected in grouped_batches(indices, batch_size, samples):
            roots = selected[::samples]
            c = model.encode(data['board'][roots], data['context'][roots])
            app, bucket = moment_losses(model.rollout_marginals(c),
                                        data['real'][selected].reshape(len(roots), samples, -1))
            count += len(roots)
            app_sum += float(app)*len(roots)
            bucket_sum += float(bucket)*len(roots)
    return dict(app_mse=app_sum/count, bucket_mse=bucket_sum/count)


def predict(model, data, indices, batch):
    """Conditional probabilities along true prefixes, for sequence likelihood only."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for selected in indices.split(batch):
            outputs.append(model(data["board"][selected], data["context"][selected],
                                 data["real"][selected]).softmax(-1))
    return torch.cat(outputs)


def trajectory_predictions(model, data, indices, batch, beam):
    model.eval()
    sequences, scores = [], []
    for selected in indices.split(batch):
        candidates, logp = model.trajectories(data["board"][selected], data["context"][selected], beam=beam)
        sequences.append(candidates)
        scores.append(logp)
    return torch.cat(sequences), torch.cat(scores)


def trajectory_metrics(sequences, logp, targets, mask):
    """Exact coverage uses complete windows; censored windows are not successes."""
    complete = mask.all(-1)
    matches = (sequences == targets[:, None]).all(-1)
    count = int(complete.sum())
    return dict(
        complete_windows=count,
        top1_exact_coverage=float(matches[complete, 0].float().mean()) if count else None,
        top3_exact_coverage=float(matches[complete].any(-1).float().mean()) if count else None,
        mean_top3_probability_mass=float(logp.exp().sum(-1).mean()),
    )


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
    model = Predictor(args.horizon, args.width, args.blocks, data["context"].shape[-1], board=use_board,
                      heads=args.heads, hidden=args.hidden, fusion=args.fusion, embedding=args.embedding)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = torch.Generator().manual_seed(args.seed)
    best_loss, best, best_epoch, stale = math.inf, None, 0, 0
    samples = args.samples if args.no_garbage else 1
    moment_weight = args.moment_weight if args.no_garbage else 0
    train_roots = torch.unique(splits['train']//samples)
    val_roots = torch.unique(splits['validation']//samples)
    history = []
    print(f"{label}: {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    for epoch in range(1, args.epochs+1):
        start = time.perf_counter()
        model.train()
        indices = train_roots[torch.randperm(len(train_roots), generator=rng)]
        loss_sum = nll_sum = app_sum = bucket_sum = 0.0
        for selected in grouped_batches(indices, args.batch_size, samples):
            roots = selected[::samples]
            root_conditioning = model.encode(data['board'][roots], data['context'][roots])
            conditioning = root_conditioning[:, None].expand(-1, samples, -1).reshape(len(selected), -1)
            logits = model.decode(conditioning, data["real"][selected])
            replay = data["replay"][selected]
            replay_logits = None
            if args.replay_weight > 0 and replay.shape[1]:
                n, replay_samples, horizon = replay.shape
                replay_logits = model.decode(
                    conditioning[:, None].expand(-1, replay_samples, -1).reshape(n*replay_samples, -1),
                    replay.reshape(n*replay_samples, horizon)).reshape(n, replay_samples, horizon, BINS)
            nll = sequence_loss(logits, data["real"][selected], data["real_mask"][selected],
                                replay, data["replay_mask"][selected], args.replay_weight, replay_logits)
            loss = nll
            if moment_weight:
                app, bucket = moment_losses(model.rollout_marginals(root_conditioning),
                                            data['real'][selected].reshape(len(roots), samples, -1))
                loss = loss + moment_weight*(app+bucket)
                app_sum += float(app.detach())*len(selected)
                bucket_sum += float(bucket.detach())*len(selected)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(selected)
            nll_sum += float(nll.detach()) * len(selected)
        val = splits["validation"]
        prob = predict(model, data, val, args.batch_size)
        val_bits = metrics(prob, data["real"][val], data["real_mask"][val])[str(args.horizon)]["bits_per_piece"]
        denom = len(splits['train'])
        validation_moments = (validate_moments(model, data, val_roots, args.batch_size, samples)
                              if moment_weight else {})
        val_loss = val_bits*math.log(2) + moment_weight*sum(validation_moments.values())
        row = dict(epoch=epoch, loss=loss_sum/denom, nll=nll_sum/denom,
                   app_mse=app_sum/denom, bucket_mse=bucket_sum/denom,
                   validation_bits=val_bits, validation_moments=validation_moments,
                   validation_objective=val_loss, seconds=time.perf_counter()-start)
        history.append(row)
        print(f"{label} epoch {epoch}: loss {row['loss']:.4f}, val {val_bits:.4f} bits/piece, "
              f"moments {validation_moments}, objective {val_loss:.4f}, {row['seconds']:.1f}s", flush=True)
        if val_loss < best_loss:
            best_loss, best, best_epoch, stale = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best)
    torch.save(dict(state_dict=best, epoch=best_epoch, args=vars(args),
                    architecture="conv-heads-gru-v1",
                    context_size=data["context"].shape[-1], board=use_board), Path(args.out)/f"{label}.pt")
    if use_board:
        directory = Path(args.out)/"heads"
        directory.mkdir(exist_ok=True)
        for i, head in enumerate(model.heads):
            torch.save(head.state_dict(), directory/f"head-{i:02d}.pt")
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
    p.add_argument('--samples', type=int, default=8,
                   help='independent trajectories per position, each with a fresh hidden queue and search run')
    p.add_argument("--thinks", type=int, default=None, help='default: 4000 with --no-garbage, otherwise 1000')
    p.add_argument("--jobs", type=int, default=None, help='default: 1 with --no-garbage, otherwise half the CPU count')
    p.add_argument("--shard-size", type=int, default=32)
    p.add_argument("--cc", default=None, help="opponent-replay binary; builds the local helper if omitted")
    p.add_argument("--replay-weight", type=float, default=0.5, help="0 skips replay generation")
    p.add_argument('--no-garbage', action='store_true',
                   help='supervise only generated no-garbage continuations; ignore recorded attacks')
    p.add_argument('--moment-weight', type=float, default=1.0,
                   help='no-garbage APP/frequency loss weight; 0 uses sequence NLL alone')
    p.add_argument('--prepare-only', action='store_true', help='cache targets without fitting models')
    p.add_argument('--exclude-positions', action='append', default=[], metavar='JSONL',
                   help='exclude roots in a saved cc_predictability run; repeatable')
    p.add_argument("--pieces", action="store_true", help="also observe hold/current/five preview pieces")
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--width", type=int, default=4, help="channels within each independent convolutional head")
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--fusion", type=int, default=256)
    p.add_argument("--hidden", type=int, default=128, help="GRU hidden size")
    p.add_argument("--embedding", type=int, default=16)
    p.add_argument("--beam", type=int, default=32, help="offline decoding beam; retain three sequences")
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
    if args.thinks is None:
        args.thinks = 4000 if args.no_garbage else 1000
    if args.jobs is None:
        args.jobs = 1 if args.no_garbage else max(1, (os.cpu_count() or 2)//2)
    if args.no_garbage and args.replay_weight <= 0:
        raise ValueError('--no-garbage requires replay generation; do not set --replay-weight 0')
    for name in ("horizon", "stride", "samples", "thinks", "jobs", "shard_size",
                 "heads", "width", "fusion", "hidden", "embedding", "epochs", "patience", "batch_size", "threads"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.moment_weight) or args.moment_weight < 0:
        raise ValueError('moment weight must be finite and nonnegative')
    if args.replay_weight < 0 or args.blocks < 0 or args.max_positions < 0:
        raise ValueError("replay weight, blocks, and position limit must be nonnegative")
    if args.beam < 3:
        raise ValueError("--beam must be at least 3")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.replay_weight and args.cc is None:
        manifest = ROOT/"scripts/opponent/opponent_replay/Cargo.toml"
        subprocess.run(["cargo", "build", "--release", "--manifest-path", str(manifest)], check=True)
        args.cc = str(manifest.parent/"target/release"/("opponent-replay.exe" if os.name == "nt" else "opponent-replay"))
    torch.set_num_threads(args.threads)
    arrays, manifest = generate(args)
    if args.prepare_only:
        examples = sum(m['examples'] for m in manifest)
        result = dict(args=vars(args), data=manifest, examples=examples,
                      source_hashes={f: digest_file(Path(__file__).parent/f) for f in
                                     ('train_opponent.py', 'opponent_data.py', 'cc_predictability.py')})
        (Path(args.out)/'data-manifest.json').write_text(json.dumps(result, indent=2))
        print(f"Prepared {examples} examples; no training started", flush=True)
        return
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
    candidates, sequence_logp = trajectory_predictions(models["cnn"], data, test, args.batch_size, args.beam)
    coverage = trajectory_metrics(candidates, sequence_logp, data["real"][test], data["real_mask"][test])
    result = dict(args=vars(args), data=manifest,
                  architecture="conv-heads-gru-v1", trajectories=coverage,
                  probability_semantics="teacher-forced conditionals; bits/piece is joint sequence NLL per observed piece",
                  split_games={k: sorted(set(arrays["game"][v])) for k, v in splits_np.items()},
                  history=histories, test=report,
                  source_hashes={f: digest_file(Path(__file__).parent/f) for f in
                                 ("train_opponent.py", "opponent_data.py", "opponent_model.py")})
    (Path(args.out)/"report.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(Path(args.out)/"test-predictions.npz", **{k: v.numpy() for k, v in probabilities.items()},
                        trajectories=candidates.numpy(), trajectory_logp=sequence_logp.numpy(),
                        targets=data["real"][test].numpy(), mask=data["real_mask"][test].numpy(),
                        games=arrays["game"][splits_np["test"]])
    print("Untouched game-split continuation targets:" if args.no_garbage else "Untouched stadium test games:", flush=True)
    print("pieces   CNN bits/piece   gain vs pooled   gain vs context", flush=True)
    for horizon, row in report["cnn"].items():
        print(f"{horizon:>6}   {row['bits_per_piece']:>14.4f}   {row['gain_vs_pooled']:>14.4f}"
              f"   {row['gain_vs_context']:>15.4f}", flush=True)
    print(f"Exact trajectory coverage (complete windows only): {coverage}", flush=True)
    print(f"Saved checkpoints, predictions and report to {args.out}", flush=True)


if __name__ == "__main__":
    main()
