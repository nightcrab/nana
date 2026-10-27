"""Score a trained predictor on saved cc_predictability rollouts, by future offset.

Open-loop marginals integrate the model's own attack prefixes by Monte Carlo.
Joint likelihood is scored exactly by teacher forcing each complete target sequence.
No model fitting or Cold Clear reruns are performed.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from cc_predictability import binned, entropy
from opponent_data import context
from opponent_model import Predictor, START


@torch.no_grad()
def open_loop(model, board, ctx, samples, seed):
    torch.manual_seed(seed)
    conditioning = model.encode(board, ctx)
    result = []
    # Encode each board once; average conditional distributions over sampled prefixes.
    for c in conditioning:
        c = c[None].expand(samples, -1)
        hidden = model.initial(c)[None].contiguous()
        previous = torch.full((samples,), START, dtype=torch.long)
        offsets = []
        for _ in range(model.horizon):
            inputs = torch.cat((model.embedding(previous), c), -1)[:, None]
            states, hidden = model.gru(inputs, hidden)
            prob = model.absorbing(model.output(states[:, 0]), previous).softmax(-1)
            offsets.append(prob.mean(0))
            previous = torch.multinomial(prob, 1).squeeze(-1)
        result.append(torch.stack(offsets))
    return torch.stack(result).numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--rollouts', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--samples', type=int, default=4096)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    settings = saved['args']
    model = Predictor(**{k: settings[k] for k in
                        ('horizon', 'width', 'blocks', 'heads', 'hidden', 'fusion', 'embedding')},
                      context_size=saved['context_size'], board=saved['board']).eval()
    model.load_state_dict(saved['state_dict'])
    with open(args.rollouts) as f:
        header = json.loads(next(f))
        rollouts = [json.loads(line) for line in f if line.strip()]
    assert header['horizon'] == model.horizon
    positions = header['positions']
    board = torch.tensor(np.asarray([p['board'][:20] for p in positions]), dtype=torch.float32)[:, None]
    ctx = torch.tensor(np.asarray([context(p, settings['pieces']) for p in positions]), dtype=torch.float32)
    targets = [[] for _ in positions]
    queues = [[] for _ in positions]
    for r in rollouts:
        if r['queue_seed'] >= 0:
            targets[r['position']].append(binned(r['moves'], model.horizon))
            queues[r['position']].append(r['queue_seed'])
    assert all(targets)
    empirical = np.asarray([[(np.bincount(np.asarray(t)[:, k], minlength=9)/len(t))
                             for k in range(model.horizon)] for t in targets])
    plugin = -(empirical * np.log2(np.maximum(empirical, 1e-300))).sum(-1)
    corrected = np.asarray([[entropy(Counter(np.asarray(t)[:, k]))
                             for k in range(model.horizon)] for t in targets])
    print(f'{len(positions)} positions, {sum(map(len, targets))} target rollouts; '
          f'{args.samples} model prefixes per position per seed', flush=True)
    estimates = []
    for seed in (1, 2):
        estimates.append(open_loop(model, board, ctx, args.samples, seed))
        print(f'Open-loop integration seed {seed} finished', flush=True)
    prediction = np.mean(estimates, axis=0)
    ce = -(empirical * np.log2(np.maximum(prediction, 1e-30))).sum(-1)
    seed_ce = [-(empirical * np.log2(np.maximum(v, 1e-30))).sum(-1).mean(0) for v in estimates]
    pooled = empirical.mean(0)
    pooled_ce = -(empirical * np.log2(np.maximum(pooled, 1e-30))).sum(-1)
    joint_nll, top3_hits, oracle_hits, held_model, held_oracle = [], [], [], [], []
    timing_hits, top10_hits = [], []
    with torch.no_grad():
        candidates, scores = model.trajectories(board, ctx, beam=128, topk=10)
        for i, target in enumerate(targets):
            t = torch.tensor(target, dtype=torch.long)
            c = model.encode(board[i:i+1], ctx[i:i+1]).expand(len(t), -1)
            logp = model.decode(c, t).log_softmax(-1).gather(-1, t[..., None]).squeeze(-1)
            joint_nll.append((-logp.sum(-1)/math.log(2)).mean().item())
            seqs = list(map(tuple, target))
            choices = set(map(tuple, candidates[i, :3].tolist()))
            choices10 = set(map(tuple, candidates[i].tolist()))
            top10_hits.append(sum(s in choices10 for s in seqs)/len(seqs))
            timing_choices = {tuple(0 < a < 8 for a in s) for s in choices}
            timing_hits.append(sum(tuple(0 < a < 8 for a in s) in timing_choices for s in seqs)/len(seqs))
            top3_hits.append(sum(s in choices for s in seqs)/len(seqs))
            oracle_hits.append(sum(n for _, n in Counter(seqs).most_common(3))/len(seqs))
            seeds = sorted(set(queues[i]))
            held_seeds = set(seeds[::4])
            training = [s for s, q in zip(seqs, queues[i]) if q not in held_seeds]
            held = [s for s, q in zip(seqs, queues[i]) if q in held_seeds]
            oracle = set(s for s, _ in Counter(training).most_common(3))
            held_model.append(sum(s in choices for s in held)/len(held))
            held_oracle.append(sum(s in oracle for s in held)/len(held))
    target_attack = empirical[:, :, 1:8].sum(-1)
    model_attack = prediction[:, :, 1:8].sum(-1)
    attack_values = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 0])
    total_error = ((prediction-empirical)*attack_values).sum((1, 2))
    joint_entropy = float(np.mean([
        -sum((n/len(t))*math.log2(n/len(t)) for n in Counter(map(tuple, t)).values())
        for t in targets]))
    offsets = [dict(piece=k+1, target_entropy_mm=float(corrected[:, k].mean()),
                    target_entropy_plugin=float(plugin[:, k].mean()),
                    model_cross_entropy=float(ce[:, k].mean()),
                    excess_over_plugin=float((ce-plugin)[:, k].mean()),
                    pooled_cross_entropy=float(pooled_ce[:, k].mean()),
                    attack_probability_mae=float(np.abs(model_attack-target_attack)[:, k].mean()),
                    attack_brier_excess=float(((model_attack-target_attack)**2)[:, k].mean()),
                    integration_seed_difference=abs(float(seed_ce[0][k]-seed_ce[1][k])))
               for k in range(model.horizon)]
    result = dict(checkpoint=args.checkpoint, rollouts=args.rollouts,
                  checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                  rollout_sha256=hashlib.sha256(Path(args.rollouts).read_bytes()).hexdigest(),
                  positions=len(positions), rollouts_count=sum(map(len, targets)),
                  samples_per_seed=args.samples, offsets=offsets,
                  joint_sequence_bits=float(np.mean(joint_nll)),
                  joint_bits_per_piece=float(np.mean(joint_nll)/model.horizon),
                  empirical_joint_entropy_bits=joint_entropy,
                  joint_empirical_kl_bits=float(np.mean(joint_nll))-joint_entropy,
                  model_top3_coverage=float(np.mean(top3_hits)),
                  model_top10_coverage=float(np.mean(top10_hits)),
                  model_top3_attack_timing_coverage=float(np.mean(timing_hits)),
                  expected_capped_total_attack_mae=float(np.abs(total_error).mean()),
                  expected_capped_total_attack_bias=float(total_error.mean()),
                  target_capped_app=float((empirical*attack_values).sum(-1).mean()),
                  model_capped_app=float((prediction*attack_values).sum(-1).mean()),
                  target_bucket_frequencies=empirical.mean(0).tolist(),
                  model_bucket_frequencies=prediction.mean(0).tolist(),
                  bucket_frequency_mse=float(((prediction-empirical)**2).sum(-1).mean()),
                  empirical_top3_coverage=float(np.mean(oracle_hits)),
                  held_queue_model_top3_coverage=float(np.mean(held_model)),
                  held_queue_empirical_top3_coverage=float(np.mean(held_oracle)),
                  model_top3_probability_mass=float(scores[:, :3].exp().sum(-1).mean()),
                  caveats=['Targets are the original no-incoming-garbage, 4000-think entropy experiment.',
                           'Training target generator, budget, and garbage policy are recorded in checkpoint args.',
                           'Model input omits bag state; entropy conditions on each complete saved position.',
                           'Marginals use Monte Carlo model prefixes; joint sequence scoring is exact.',
                           'Empirical entropies and top-three ceilings use only 64 target samples per position.'])
    Path(args.out).write_text(json.dumps(result, indent=2))
    print('piece  target H(MM)  model open-loop CE  pooled CE  MC seed difference')
    for r in offsets:
        print(f"{r['piece']:5} {r['target_entropy_mm']:13.4f} {r['model_cross_entropy']:19.4f} "
              f"{r['pooled_cross_entropy']:10.4f} {r['integration_seed_difference']:19.4f}")
    print(json.dumps({k:v for k,v in result.items() if k not in ('offsets','caveats')}, indent=2))


if __name__ == '__main__':
    main()
