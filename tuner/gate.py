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


def score_sprt(scores, p1, alpha, beta):
    """One-sample Bernoulli score LLR against p0=0.5; draws count half.

    Fractional draw scores use a half-win/half-loss likelihood contribution.
    Unlike the Gaussian gate, this does not estimate variance from the sample.
    """
    llr = sum(score * math.log(p1 / 0.5) + (1 - score) * math.log((1 - p1) / 0.5)
              for score in scores)
    if llr >= math.log((1 - beta) / alpha):
        return llr, "accept"
    if llr <= math.log(beta / (1 - alpha)):
        return llr, "reject"
    return llr, "continue"


def run_stadium_gate(runner, stage, candidate, release, seed0, delta, alpha, beta, max_games, batch, log):
    p1 = 0.5 * (1 + delta)
    if not 0 < delta < 1:
        raise ValueError("stadium gate delta must be in (0, 1)")
    if not 0 < alpha < 1 or not 0 < beta < 1 or alpha + beta >= 1:
        raise ValueError("stadium gate alpha and beta must be positive and sum to less than 1")
    if max_games < 1 or batch < 1:
        raise ValueError("stadium gate max_games and batch must be positive")
    scores = []
    seed = seed0
    llr, verdict = 0.0, "continue"
    # The release is the opponent, including when it differs from the tuning baseline.
    # The context restores the tuning opponent on success, failure, or interruption.
    with runner.against(release):
        while len(scores) < max_games:
            n = min(batch, max_games - len(scores))
            values, _ = runner.scores([("cand", candidate, stage, seed, n)])
            scores += values["cand"]
            seed += n
            llr, verdict = score_sprt(scores, p1, alpha, beta)
            log(f"  stadium gate: {len(scores)} games vs release, candidate {statistics.mean(scores):.4f}, "
                f"null 0.5000, alternative {p1:.4f}, LLR {llr:.2f}")
            if verdict != "continue":
                break
    if verdict == "continue":
        verdict = "inconclusive"
    return {"verdict": verdict, "games": len(scores), "llr": llr, "candidate": statistics.mean(scores),
            "release": 0.5, "null_score": 0.5, "alternative_score": p1,
            "method": "stadium-score-sprt", "opponent": "release",
            "wins": scores.count(1.0), "draws": scores.count(0.5), "losses": scores.count(0.0),
            "seed0": seed0, "seed_end": seed, "stage": stage.spec}


def run_gate(runner, stage, candidate, release, seed0, delta=0.02, alpha=0.05, beta=0.05, max_games=2000,
             batch=None, log=print):
    batch = batch or 4 * runner.workers
    if stage.objective == "stadium":
        return run_stadium_gate(runner, stage, candidate, release, seed0, delta, alpha, beta,
                                max_games, batch, log)
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
