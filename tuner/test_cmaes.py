import json
import math
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .cmaes import CMAESOptimizer
from .optimizer import create_optimizer, optimizer_defaults, restore_optimizer
from .run import Run
from .runner import Runner


class CMAESTests(unittest.TestCase):
    def test_rotated_objective_learns_correlations_and_improves(self):
        optimizer = CMAESOptimizer([0.5, 0.5], 12, 0.2, 4)

        def objective(unit):
            x, y = np.array(unit) - [0.6, 0.4]
            return -(1000 * (x + y) ** 2 + (x - y) ** 2)

        for _ in range(80):
            optimizer.tell({c.id: [objective(c.unit)] for c in optimizer.ask()})
        self.assertGreater(optimizer.best().mean, -1e-6)
        covariance = optimizer.covariance
        self.assertLess(covariance[0, 1] / math.sqrt(covariance[0, 0] * covariance[1, 1]), -0.8)
        self.assertTrue((np.linalg.eigvalsh(covariance) > 0).all())

    def test_json_resume_is_exact_and_ask_is_read_only(self):
        settings = {**optimizer_defaults("cmaes"), "optimizer": "cmaes", "seed": 0}
        optimizer = create_optimizer(settings, [0.1, 0.8, 0.4])
        for generation in range(8):
            saved = json.loads(json.dumps({"optimizer": optimizer.state()}, allow_nan=False))
            restored = restore_optimizer(settings, saved)
            points = optimizer.ask()
            self.assertEqual(points, restored.ask())
            points[0].unit[0] = -100  # No alias into the cached population.
            self.assertEqual(saved["optimizer"], optimizer.state())
            scores = {c.id: [-sum((u - 0.6) ** 2 for u in c.unit) + offset + generation
                             for offset in (-0.1, 0.1)] for c in optimizer.ask()}
            self.assertEqual(optimizer.tell(scores), restored.tell(scores))
            self.assertEqual(optimizer.state(), restored.state())
            self.assertEqual(optimizer.best(), restored.best())

    def test_promotion_uses_fresh_paired_incumbent_not_stale_mean(self):
        optimizer = CMAESOptimizer([0.5], 2, 0.3, 1)
        self.assertIsNone(optimizer.best())
        winner = optimizer.ask()[0].unit
        feedback = optimizer.tell({"c0": [100.0], "c1": [0.0], "incumbent": [99.0]})
        self.assertEqual(feedback.best_id, "c0")
        self.assertEqual(optimizer.best().unit, winner)
        feedback = optimizer.tell({"c0": [2.0], "c1": [1.0], "incumbent": [3.0]})
        self.assertEqual(feedback.best_id, "incumbent")
        self.assertEqual(feedback.best_latest, 3.0)
        self.assertEqual(optimizer.best().mean, 51.5)
        self.assertEqual(optimizer.best().evaluations, 2)
        winner = optimizer.ask()[1].unit
        feedback = optimizer.tell({"c0": [2.0], "c1": [4.0], "incumbent": [3.0]})
        self.assertEqual(feedback.best_id, "c1")
        self.assertEqual(optimizer.best().unit, winner)
        self.assertEqual(optimizer.best().mean, 4.0)
        self.assertEqual(optimizer.best().evaluations, 1)
        self.assertEqual(optimizer.evaluation_counts(3), [3, 3, 3])

    def test_bounds_and_reproducibility_at_boundary_centers(self):
        for center in ([0.0], [1.0], [0.0, 1.0, 0.5]):
            optimizer = CMAESOptimizer(center, 6, 2.0, 0)
            other = CMAESOptimizer(center, 6, 2.0, 0)
            for _ in range(12):
                self.assertEqual(optimizer.ask(), other.ask())
                for candidate in optimizer.ask():
                    self.assertTrue(all(0 <= u <= 1 for u in candidate.unit))
                scores = {c.id: [-sum(u * u for u in c.unit)] for c in optimizer.ask()}
                optimizer.tell(scores)
                other.tell(scores)

    def test_invalid_options_and_scores_fail_before_advancing(self):
        for center, population, spread in (([], 4, 0.3), ([float("nan")], 4, 0.3),
                                           ([-0.1], 4, 0.3), ([0.5], 1, 0.3),
                                           ([0.5], 2.5, 0.3), ([0.5], 4, 0),
                                           ([0.5], 4, float("inf"))):
            with self.subTest(center=center, population=population, spread=spread):
                with self.assertRaises(ValueError):
                    CMAESOptimizer(center, population, spread, 1)
        optimizer = CMAESOptimizer([0.5], 2, 0.3, 1)
        saved = optimizer.state()
        for scores in ({}, {"c0": [], "c1": [1], "incumbent": [1]},
                       {"c0": [float("nan")], "c1": [1], "incumbent": [1]},
                       {"c0": [1, 2], "c1": [1], "incumbent": [1]}):
            with self.assertRaises(ValueError):
                optimizer.tell(scores)
            self.assertEqual(saved, optimizer.state())

    def test_run_calibration_resume_holdout_and_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = os.path.join(directory, "nana")
            with open(binary, "w"):
                pass
            space = os.path.join(directory, "space.json")
            with open(space, "w") as f:
                json.dump([{"name": "weight", "default": 0.2, "low": 0, "high": 1}], f)

            def spawn(runner, cmd, env):
                with open(env["NANA_CONFIG"]) as f:
                    config = json.load(f)
                count = int(cmd[cmd.index("--games") + 1])
                game = {"rules": "tetrio", "attack": 100 * config.get("weight", 0.2), "died": False,
                        "sims_per_move": 128, "ms_per_move": 1 / (1 + config.get("weight", 0.2))}
                return 0, "\n".join(json.dumps(game) for _ in range(count)), ""

            with patch.object(Runner, "print_config", return_value={"weight": 0.2}), \
                    patch.object(Runner, "_spawn", new=spawn), patch.object(Run, "health_checks"):
                for objective in ("strength", "sims-per-second"):
                    run = Run.create(os.path.join(directory, objective), binary, [0], [space],
                                     {"optimizer": "cmaes", "objective": objective, "sims": 128,
                                      "particles": 4, "iterations": 4, "games_per_eval": 2, "gate_games": 8})
                    self.assertNotIn("pso", run.settings)
                    run.run(plan_only=True)
                    run.baseline()
                    run.iterate()
                    run.save()
                    resumed = Run(run.dir)
                    self.assertEqual(run.optimizer.state(), resumed.optimizer.state())
                    resumed.run()
                    final = Run(run.dir)
                    self.assertEqual(final.state["phase"], "done")
                    self.assertEqual(final.optimizer.name, "cmaes")
                    self.assertEqual(final.optimizer.iteration, 4)
                    with open(final.path("evals.jsonl")) as f:
                        records = [json.loads(line) for line in f]
                    self.assertEqual(len(records), 4)
                    self.assertTrue(all(r["optimizer"] == "cmaes" and len(r["scores"]) == 5 for r in records))
                    self.assertEqual(len(final.state["holdouts"]), 1)
                    self.assertGreaterEqual(final.optimizer.best().mean, 0)


if __name__ == "__main__":
    unittest.main()
