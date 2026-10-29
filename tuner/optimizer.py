"""Optimizer contract: maximize scores in normalized [0, 1] parameter space.

Candidate IDs are opaque strings. All candidates in an ask batch receive the same
fresh game seeds, including any reevaluations requested by the optimizer.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Candidate:
    id: str
    unit: list


@dataclass
class Recommendation:
    index: object
    unit: list
    mean: float
    evaluations: int


@dataclass
class Feedback:
    candidate_scores: list
    best_latest: float | None
    best_id: str


class Optimizer(ABC):
    name: str

    @abstractmethod
    def ask(self) -> list[Candidate]:
        """Return a batch without advancing the optimizer or consuming randomness."""

    @abstractmethod
    def tell(self, scores: dict[str, list[float]]) -> Feedback:
        """Consume per-game scores, advance once, and identify best's evidence."""

    @abstractmethod
    def best(self) -> Recommendation | None:
        pass

    @abstractmethod
    def sample_points(self) -> list:
        """Representative unit vectors for calibration and config checks."""

    @abstractmethod
    def evaluation_counts(self, iterations: int) -> list[int]:
        """Expected candidate counts for a complete run, for budget projection."""

    @abstractmethod
    def diagnostics(self) -> dict:
        """JSON-compatible algorithm metrics for logs and status."""

    @abstractmethod
    def warnings(self, parameter_names: list[str]) -> list[str]:
        pass

    @abstractmethod
    def state(self) -> dict:
        """Versioned JSON checkpoint; immutable settings live in run.json."""


OPTIMIZERS = ("pso", "cmaes", "spsa")


def optimizer_defaults(name):
    from .cmaes import CMAES_DEFAULTS
    from .pso import PSO_DEFAULTS
    from .spsa import SPSA_DEFAULTS

    if name == "spsa":
        return SPSA_DEFAULTS
    if name == "cmaes":
        return CMAES_DEFAULTS
    if name != "pso":
        raise ValueError(f"unknown optimizer: {name}")
    return PSO_DEFAULTS


def create_optimizer(settings, center):
    from .cmaes import CMAESOptimizer
    from .pso import PSOOptimizer
    from .spsa import SPSAOptimizer

    name = settings.get("optimizer", "pso")
    if name == "spsa":
        return SPSAOptimizer.create(settings, center)
    if name == "cmaes":
        return CMAESOptimizer.create(settings, center)
    if name != "pso":
        raise ValueError(f"unknown optimizer: {name}")
    return PSOOptimizer.create(settings, center)


def restore_optimizer(settings, state):
    from .cmaes import CMAESOptimizer
    from .pso import PSOOptimizer
    from .spsa import SPSAOptimizer

    # Version-2 runs stored the bare PSO state under `pso`.
    saved = state.get("optimizer")
    if saved is None:
        saved = {"name": "pso", "version": 1, "state": state["pso"]}
    name = settings.get("optimizer", "pso")
    if saved["name"] != name:
        raise ValueError("optimizer setting does not match checkpoint")
    if name not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer: {name}")
    if saved["version"] != 1:
        raise ValueError(f"unsupported {name} optimizer state version: {saved['version']}")
    if name == "cmaes":
        return CMAESOptimizer.from_state(settings, saved["state"])
    if name == "spsa":
        return SPSAOptimizer.from_state(settings, saved["state"])
    return PSOOptimizer.from_state(settings, saved["state"])
