import math
import statistics

# Make the failure obvious even at the health check's small search budget.
# Mildly reversing a few board weights can still play well enough to pass short games.
CANARY = {
    "eval": "nana",
    "height": 10000.0,
    "top_half": 10000.0,
    "top_quarter": 10000.0,
    "cavity_cells": 10000.0,
    "cavity_cells_sq": 1000.0,
    "covered_cells": 1000.0,
    "bumpiness": 1000.0,
    "bumpiness_sq": 1000.0,
    "row_transitions": 1000.0,
    "clears": [0, -100000.0, -100000.0, -100000.0, -100000.0],
    "tspins": [0, 0, 0, 0],
    "perfect_clear": 0.0,
    "wasted_t": 0.0,
}


class HealthError(RuntimeError):
    pass


def _close(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)))


def check_plumbing(runner, configs):
    for config in configs:
        loaded = runner.print_config(config)
        for key, value in config.items():
            if key not in loaded:
                raise HealthError(f"plumbing: nana doesn't know the key {key}")
            if not _close(value, loaded[key]):
                raise HealthError(f"plumbing: {key} written as {value}, nana loaded {loaded[key]}")


def check_determinism(runner, stage, config, seed0, games):
    a, _ = runner.scores([("a", config, stage, seed0, games)])
    b, _ = runner.scores([("b", config, stage, seed0, games)])
    if a["a"] != b["b"]:
        raise HealthError(f"determinism: the same config on the same seeds scored {a['a']} then {b['b']}")


def check_canary(runner, stage, base, seed0, games):
    canary = dict(base)
    canary.update(CANARY)
    scores, _ = runner.scores([("base", base, stage, seed0, games), ("canary", canary, stage, seed0, games)])
    diffs = [b - c for b, c in zip(scores["base"], scores["canary"])]
    mean = statistics.mean(diffs)
    se = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else float("inf")
    if not mean > 3 * se:
        raise HealthError(f"canary: the bad config isn't clearly worse (baseline - canary = {mean:.4f} +- {se:.4f}); "
                          "the config may not reach the search, or the fitness may not respond")
    return statistics.mean(scores["base"]), statistics.mean(scores["canary"])


class Monitor:
    def __init__(self):
        self.curse_gaps = []

    def iteration(self, pso, space, scores_x, game_sd, games, best_latest, best_mean, best_games, baseline):
        warnings = []
        noise = game_sd / math.sqrt(games) if games else float("inf")
        spread = statistics.pstdev(scores_x) if len(scores_x) > 1 else 0.0
        if spread < 1.5 * noise:
            warnings.append(f"signal: the swarm's scores spread {spread:.4f}, about one evaluation's noise "
                            f"({noise:.4f}); more games per evaluation would help")
        if best_latest is not None:
            self.curse_gaps = (self.curse_gaps + [best_mean - best_latest])[-10:]
            if len(self.curse_gaps) >= 5 and statistics.mean(self.curse_gaps) > 2 * noise:
                warnings.append(f"winner's curse: the best's fresh scores run {statistics.mean(self.curse_gaps):.4f} "
                                "below its average; selection may be chasing noise")
        if pso.diversity() < 0.01:
            warnings.append(f"collapse: the swarm has converged (diversity {pso.diversity():.4f})")
        pinned = [space.names()[d] for d, share in enumerate(pso.at_bounds()) if share >= 0.5]
        if pinned:
            warnings.append("bounds: most bests sit on a wall for " + ", ".join(pinned[:8])
                            + ("..." if len(pinned) > 8 else "") + "; widen the bounds or check for an exploit")
        if best_games and baseline:
            died = statistics.mean(g["died"] for g in best_games)
            if died > baseline["died"] + 0.25:
                warnings.append(f"behaviour: the best dies in {died:.0%} of games against the baseline's "
                                f"{baseline['died']:.0%}")
        return warnings
