"""Positive-weight (mu/mu_w, lambda)-CMA-ES with cumulative step-size adaptation.

Algorithm and default learning rates: Hansen, The CMA Evolution Strategy: A
Tutorial, https://arxiv.org/abs/1604.00772 . The Gaussian lives in unbounded
coordinates; a reflection transform maps evaluated points into the unit cube.
"""
import math
import statistics
from dataclasses import asdict

import numpy as np

from .optimizer import Candidate, Feedback, Optimizer, Recommendation

CMAES_DEFAULTS = {"particles": 24, "spread": 0.3}


def _reflect(points):
    return 1.0 - np.abs(np.remainder(points, 2.0) - 1.0)


class CMAESOptimizer(Optimizer):
    name = "cmaes"

    def __init__(self, center, population, spread, seed):
        center = np.asarray(center, dtype=float)
        if center.ndim != 1 or not len(center) or not np.isfinite(center).all():
            raise ValueError("CMA-ES needs a nonempty finite parameter vector")
        if np.any((center < 0) | (center > 1)):
            raise ValueError("CMA-ES center must be in [0, 1]")
        if not isinstance(population, int) or population < 2:
            raise ValueError("CMA-ES particles must be an integer of at least 2")
        if not math.isfinite(spread) or spread <= 0:
            raise ValueError("CMA-ES spread must be finite and positive")
        self.dim = len(center)
        self.population = population
        self.mean = center.copy()
        self.sigma = float(spread)
        self.covariance = np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.iteration = 0
        self.rng = np.random.default_rng(seed)
        self.incumbent = Recommendation("defaults", center.tolist(), 0.0, 0)
        self._parameters()
        self._sample()

    def _parameters(self):
        n = self.dim
        self.mu = self.population // 2
        weights = np.log((self.population + 1) / 2) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / weights.sum()
        self.mueff = 1 / np.sum(self.weights ** 2)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff)
                       / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chi = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

    @classmethod
    def create(cls, settings, center):
        return cls(center, settings["particles"], settings["spread"], settings.get("seed", 1))

    def _sample(self):
        # Eigh of the symmetric covariance provides both sampling and whitening.
        eigenvalues, self.basis = np.linalg.eigh(self.covariance)
        self.scales = np.sqrt(np.maximum(eigenvalues, 1e-30))
        self.covariance = (self.basis * self.scales ** 2) @ self.basis.T
        z = self.rng.standard_normal((self.population, self.dim))
        self.steps = (z * self.scales) @ self.basis.T
        self.points = _reflect(self.mean + self.sigma * self.steps)

    def ask(self):
        # Sampling happens at creation/after tell, so repeated asks are idempotent.
        return [Candidate(f"c{i}", point.tolist()) for i, point in enumerate(self.points)] + [
            Candidate("incumbent", list(self.incumbent.unit))]

    def tell(self, scores):
        candidates = self.ask()
        if set(scores) != {c.id for c in candidates}:
            raise ValueError("CMA-ES scores must match the requested candidate IDs")
        if any(not values or not all(math.isfinite(v) for v in values) for values in scores.values()):
            raise ValueError("CMA-ES needs nonempty finite game scores")
        if len({len(values) for values in scores.values()}) != 1:
            raise ValueError("CMA-ES candidates must use the same number of paired games")
        means = np.array([statistics.mean(scores[f"c{i}"]) for i in range(self.population)])
        order = np.argsort(-means, kind="stable")  # maximize
        winner = int(order[0])
        incumbent_score = statistics.mean(scores["incumbent"])
        latest = incumbent_score if self.incumbent.evaluations else None
        if means[winner] > incumbent_score:
            self.incumbent = Recommendation(f"{self.iteration}:c{winner}", self.points[winner].tolist(),
                                            float(means[winner]), 1)
            evidence = f"c{winner}"
        else:
            best = self.incumbent
            best.mean = (best.mean * best.evaluations + incumbent_score) / (best.evaluations + 1)
            best.evaluations += 1
            evidence = "incumbent"

        selected = self.steps[order[:self.mu]]
        step = self.weights @ selected
        self.mean += self.sigma * step
        whitened = self.basis @ ((self.basis.T @ step) / self.scales)
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * whitened
        ps_norm = float(np.linalg.norm(self.ps))
        hsig = ps_norm / math.sqrt(1 - (1 - self.cs) ** (2 * (self.iteration + 1))) < (
            1.4 + 2 / (self.dim + 1)) * self.chi
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * step
        self.covariance = (
            (1 - self.c1 - self.cmu + self.c1 * (not hsig) * self.cc * (2 - self.cc)) * self.covariance
            + self.c1 * np.outer(self.pc, self.pc)
            + self.cmu * (selected.T * self.weights) @ selected)
        self.covariance = (self.covariance + self.covariance.T) / 2
        self.sigma *= math.exp(min(1.0, self.cs / self.damps * (ps_norm / self.chi - 1)))
        self.iteration += 1
        self._sample()
        return Feedback(means.tolist(), latest, evidence)

    def best(self):
        return Recommendation(**asdict(self.incumbent)) if self.incumbent.evaluations else None

    def sample_points(self):
        return [list(self.incumbent.unit)] + self.points.tolist()

    def evaluation_counts(self, iterations):
        return [self.population + 1] * iterations

    def diagnostics(self):
        return {"sigma": self.sigma, "condition": float((self.scales[-1] / self.scales[0]) ** 2),
                "diversity": float(np.mean(np.linalg.norm(self.points - self.points.mean(axis=0), axis=1))
                                   / math.sqrt(self.dim))}

    def warnings(self, parameter_names):
        warnings = []
        if self.sigma * self.scales[-1] < 1e-8:
            warnings.append("collapse: CMA-ES search steps have converged below 1e-8 unit-cube units")
        if self.scales[-1] / self.scales[0] > 1e7:
            warnings.append("condition: CMA-ES covariance is ill-conditioned; consider restarting")
        if self.incumbent.evaluations:
            pinned = [name for name, value in zip(parameter_names, self.incumbent.unit)
                      if value < 0.01 or value > 0.99]
            if pinned:
                warnings.append("bounds: the best is near a limit for " + ", ".join(pinned[:8])
                                + ("..." if len(pinned) > 8 else "") + "; check the bounds")
        return warnings

    def state(self):
        return {"name": self.name, "version": 1, "state": {
            "population": self.population, "mean": self.mean.tolist(), "sigma": self.sigma,
            "covariance": self.covariance.tolist(), "pc": self.pc.tolist(), "ps": self.ps.tolist(),
            "iteration": self.iteration, "rng": self.rng.bit_generator.state,
            "incumbent": asdict(self.incumbent), "basis": self.basis.tolist(),
            "scales": self.scales.tolist(), "steps": self.steps.tolist(), "points": self.points.tolist()}}

    @classmethod
    def from_state(cls, settings, state):
        # Restore cached samples and their factorization directly: no resampling
        # or eigendecomposition, so JSON round trips preserve the trajectory.
        optimizer = cls.__new__(cls)
        optimizer.dim = len(state["mean"])
        optimizer.population = state["population"]
        optimizer.sigma = state["sigma"]
        optimizer.iteration = state["iteration"]
        for field in ("mean", "covariance", "pc", "ps", "basis", "scales", "steps", "points"):
            setattr(optimizer, field, np.array(state[field], dtype=float))
        optimizer.incumbent = Recommendation(**state["incumbent"])
        optimizer.rng = np.random.default_rng()
        optimizer.rng.bit_generator.state = state["rng"]
        optimizer._parameters()
        return optimizer
