"""Release gates: does a candidate beat the current release? A sequential probability ratio test on
paired games (the same seeds for both), so it stops as soon as the answer is clear.

H0: the candidate's mean fitness is no better (difference <= 0)
H1: it's better by at least `delta` (a fraction of the release's fitness, e.g. 0.02 = 2%)
With d the paired per-game differences and s^2 their variance, the log-likelihood ratio under a
normal model is LLR = (mu1 - mu0) / s^2 * (sum d - n (mu0 + mu1) / 2). Accept H1 when LLR passes
log((1 - beta) / alpha), H0 when it falls below log(beta / (1 - alpha)). At max_games without an
answer the candidate isn't released (the release only changes on evidence).
"""

import math
import statistics


def sprt(diffs, mu1, alpha, beta):
    n = len(diffs)
    if n < 8:
        return 0.0, "continue"
    var = statistics.variance(diffs)
    if var <= 0:
        var = 1e-12
    llr = (mu1 - 0.0) / var * (sum(diffs) - n * (0.0 + mu1) / 2)
    upper = math.log((1 - beta) / alpha)
    lower = math.log(beta / (1 - alpha))
    if llr >= upper:
        return llr, "accept"
    if llr <= lower:
        return llr, "reject"
    return llr, "continue"


def run_gate(runner, stage, candidate, release, seed0, delta=0.02, alpha=0.05, beta=0.05, max_games=2000,
             batch=None, log=print):
    """Plays candidate and release on the same seeds in batches until the SPRT decides. Returns a
    dict with the verdict ('accept', 'reject' or 'inconclusive') and the evidence."""
    batch = batch or 4 * len(runner.cpus)
    cand, rel = [], []
    verdict, llr = "continue", 0.0
    seed = seed0
    while len(cand) < max_games:
        n = min(batch, max_games - len(cand))
        scores, _ = runner.scores([("cand", candidate, stage, seed, n), ("rel", release, stage, seed, n)])
        cand += scores["cand"]
        rel += scores["rel"]
        seed += n
        release_mean = statistics.mean(rel)
        mu1 = max(delta * abs(release_mean), 1e-6)
        llr, verdict = sprt([c - r for c, r in zip(cand, rel)], mu1, alpha, beta)
        log(f"  gate: {len(cand)} games, candidate {statistics.mean(cand):.4f} vs release {release_mean:.4f}, "
            f"LLR {llr:.2f}")
        if verdict != "continue":
            break
    if verdict == "continue":
        verdict = "inconclusive"
    return {"verdict": verdict, "games": len(cand), "llr": llr, "candidate": statistics.mean(cand),
            "release": statistics.mean(rel), "seed0": seed0, "seed_end": seed, "stage": stage.spec}
