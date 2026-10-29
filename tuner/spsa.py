"""Projected simultaneous perturbation stochastic approximation (maximization)."""
import math
import statistics
from dataclasses import asdict

import numpy as np

from .optimizer import Candidate, Feedback, Optimizer, Recommendation

SPSA_DEFAULTS = {
    "spsa_learning_rate": 0.05,
    "spsa_perturbation": 0.1,
    "spsa_stability": 10.0,
    "spsa_alpha": 0.602,
    "spsa_gamma": 0.101,
    "spsa_max_step": 0.1,
    "spsa_start_jitter": 0.05,
}

# Options that may be zero; every other option must be positive.
_NONNEGATIVE = ("spsa_stability", "spsa_start_jitter")


class SPSAOptimizer(Optimizer):
    name = "spsa"

    def __init__(self, center, options, seed, jitter=True):
        self.center = np.asarray(center, dtype=float).copy()
        if self.center.ndim != 1 or not len(self.center) or not np.isfinite(self.center).all():
            raise ValueError("SPSA needs a nonempty finite parameter vector")
        if np.any((self.center < 0) | (self.center > 1)):
            raise ValueError("SPSA center must be in [0, 1]")
        self.options = {key: options.get(key, default) for key, default in SPSA_DEFAULTS.items()}
        for key, value in self.options.items():
            if not math.isfinite(value) or (value < 0 if key in _NONNEGATIVE else value <= 0):
                raise ValueError(f"{key} must be finite and " +
                                 ("nonnegative" if key in _NONNEGATIVE else "positive"))
        for key in ("spsa_perturbation", "spsa_start_jitter"):
            if self.options[key] > 0.5:
                raise ValueError(f"{key} must be at most 0.5")
        if self.options["spsa_max_step"] > 1:
            raise ValueError("spsa_max_step must be at most 1")
        for key in ("spsa_alpha", "spsa_gamma"):
            if self.options[key] > 1:
                raise ValueError(f"{key} must be at most 1")
        self.iteration = 0
        self.rng = np.random.default_rng(seed)
        self.incumbent = Recommendation("defaults", self.center.tolist(), 0.0, 0)
        if jitter:
            # Mirrored probes cancel every even term of the objective, so a center where the gradient is zero
            # (such as the all-zero point of a product of weights) never moves. Starting a random offset away
            # avoids that; the incumbent stays at the defaults.
            signs = self.rng.choice([-1.0, 1.0], size=len(self.center))
            self.center = np.clip(self.center + self.options["spsa_start_jitter"] * signs, 0, 1)
        self._sample()

    @classmethod
    def create(cls, settings, center):
        return cls(center, settings, settings.get("seed", 1))

    def _gains(self):
        k = self.iteration + 1
        o = self.options
        return (o["spsa_learning_rate"] / (k + o["spsa_stability"]) ** o["spsa_alpha"],
                o["spsa_perturbation"] / k ** o["spsa_gamma"])

    def _sample(self):
        _, perturbation = self._gains()
        direction = self.rng.choice([-1.0, 1.0], size=len(self.center))
        self.plus = np.clip(self.center + perturbation * direction, 0, 1)
        self.minus = np.clip(self.center - perturbation * direction, 0, 1)

    def ask(self):
        return [Candidate("plus", self.plus.tolist()), Candidate("minus", self.minus.tolist()),
                Candidate("incumbent", list(self.incumbent.unit))]

    def tell(self, scores):
        if set(scores) != {"plus", "minus", "incumbent"}:
            raise ValueError("SPSA scores must match the requested candidate IDs")
        if any(not values or not all(math.isfinite(v) for v in values) for values in scores.values()):
            raise ValueError("SPSA needs nonempty finite game scores")
        if len({len(values) for values in scores.values()}) != 1:
            raise ValueError("SPSA candidates must use the same number of paired games")
        means = {key: statistics.mean(values) for key, values in scores.items()}
        latest = means["incumbent"] if self.incumbent.evaluations else None
        winner = max(("plus", "minus"), key=means.get)
        if means[winner] > means["incumbent"]:
            point = self.plus if winner == "plus" else self.minus
            self.incumbent = Recommendation(f"{self.iteration}:{winner}", point.tolist(), means[winner], 1)
            evidence = winner
        else:
            best = self.incumbent
            best.mean = (best.mean * best.evaluations + means["incumbent"]) / (best.evaluations + 1)
            best.evaluations += 1
            evidence = "incumbent"

        # Use the actual signed separation to account for clipped boundary probes.
        gradient = (means["plus"] - means["minus"]) / (self.plus - self.minus)
        rate, _ = self._gains()
        step = np.clip(rate * gradient, -self.options["spsa_max_step"], self.options["spsa_max_step"])
        self.center = np.clip(self.center + step, 0, 1)
        self.iteration += 1
        self._sample()
        return Feedback([means["plus"], means["minus"]], latest, evidence)

    def best(self):
        return Recommendation(**asdict(self.incumbent)) if self.incumbent.evaluations else None

    def sample_points(self):
        return [self.center.tolist(), self.plus.tolist(), self.minus.tolist()]

    def evaluation_counts(self, iterations):
        return [3] * iterations

    def diagnostics(self):
        rate, perturbation = self._gains()
        return {"learning_rate": rate, "perturbation": perturbation}

    def warnings(self, parameter_names):
        pinned = [name for name, value in zip(parameter_names, self.center)
                  if value < 0.01 or value > 0.99]
        if pinned:
            return ["bounds: SPSA center is near a limit for " + ", ".join(pinned[:8])
                    + ("..." if len(pinned) > 8 else "") + "; check the bounds"]
        return []

    def state(self):
        return {"name": self.name, "version": 1, "state": {
            "center": self.center.tolist(), "iteration": self.iteration,
            "plus": self.plus.tolist(), "minus": self.minus.tolist(),
            "incumbent": asdict(self.incumbent), "rng": self.rng.bit_generator.state}}

    @classmethod
    def from_state(cls, settings, state):
        optimizer = cls(state["center"], settings, 0, jitter=False)
        optimizer.iteration = state["iteration"]
        optimizer.plus = np.array(state["plus"], dtype=float)
        optimizer.minus = np.array(state["minus"], dtype=float)
        optimizer.incumbent = Recommendation(**state["incumbent"])
        optimizer.rng.bit_generator.state = state["rng"]
        return optimizer
