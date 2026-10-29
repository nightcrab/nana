import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .optimizer import create_optimizer, optimizer_defaults, restore_optimizer
from .run import Run
from .runner import Runner


class SPSATests(unittest.TestCase):
    def settings(self, **options):
        return {**optimizer_defaults("spsa"), "optimizer": "spsa", "seed": 7, **options}

    def test_gradient_ascent_and_boundary_separation(self):
        settings = self.settings(spsa_learning_rate=0.02, spsa_stability=0, spsa_start_jitter=0)
        for center in ([0.4], [0.0], [1.0]):
            optimizer = create_optimizer(settings, center)
            before = optimizer.center.copy()
            optimizer.tell({c.id: [2 * c.unit[0]] for c in optimizer.ask()})
            np.testing.assert_allclose(optimizer.center, np.clip(before + 0.04, 0, 1))
        optimizer = create_optimizer(self.settings(spsa_learning_rate=100, spsa_start_jitter=0), [0.4])
        optimizer.tell({c.id: [c.unit[0]] for c in optimizer.ask()})
        np.testing.assert_allclose(optimizer.center, [0.5])  # max-step cap

    def test_start_jitter(self):
        defaults = [0.5, 0.0, 1.0, 0.3]
        optimizer = create_optimizer(self.settings(spsa_start_jitter=0.2), defaults)
        offset = optimizer.center - np.array(defaults)
        np.testing.assert_allclose(np.abs(offset[[0, 3]]), 0.2)
        self.assertIn(offset[1], (0.0, 0.2))
        self.assertIn(offset[2], (0.0, -0.2))
        self.assertTrue(all(0 <= u <= 1 for u in optimizer.center))
        self.assertEqual(optimizer.ask()[2].unit, defaults)
        unjittered = create_optimizer(self.settings(spsa_start_jitter=0), defaults)
        np.testing.assert_array_equal(unjittered.center, defaults)

    def test_start_jitter_escapes_product_saddle(self):
        def objective(unit):
            return (unit[0] - 0.5) * (unit[1] - 0.5)
        for jitter, escapes in ((0, False), (0.05, True)):
            optimizer = create_optimizer(self.settings(spsa_start_jitter=jitter, spsa_learning_rate=1), [0.5, 0.5])
            for _ in range(200):
                optimizer.tell({c.id: [objective(c.unit)] for c in optimizer.ask()})
            if escapes:
                self.assertGreater(objective(optimizer.center), 0.05)
            else:
                np.testing.assert_array_equal(optimizer.center, [0.5, 0.5])

    def test_improves_multidimensional_objective(self):
        optimizer = create_optimizer(self.settings(), [0.2, 0.8, 0.3])
        def objective(unit):
            return -sum((x - 0.6) ** 2 for x in unit)
        initial = objective([0.2, 0.8, 0.3])
        for _ in range(1000):
            optimizer.tell({c.id: [objective(c.unit)] for c in optimizer.ask()})
        self.assertGreater(optimizer.best().mean, initial + 0.25)
        self.assertLess(np.linalg.norm(optimizer.center - 0.6), 0.06)

    def test_json_resume_exact_and_no_aliases(self):
        settings = self.settings()
        optimizer = create_optimizer(settings, [0.0, 0.8, 1.0])
        for _ in range(12):
            saved = json.loads(json.dumps({"optimizer": optimizer.state()}, allow_nan=False))
            restored = restore_optimizer(settings, saved)
            self.assertEqual(optimizer.ask(), restored.ask())
            optimizer.ask()[0].unit[0] = -100
            optimizer.sample_points()[0][0] = -100
            if optimizer.best():
                optimizer.best().unit[0] = -100
            self.assertEqual(saved["optimizer"], optimizer.state())
            for candidate in optimizer.ask():
                self.assertTrue(all(0 <= u <= 1 for u in candidate.unit))
            scores = {c.id: [-sum((u - 0.6) ** 2 for u in c.unit) + offset
                             for offset in (-0.1, 0.1)] for c in optimizer.ask()}
            self.assertEqual(optimizer.tell(scores), restored.tell(scores))
            self.assertEqual(optimizer.state(), restored.state())

    def test_fresh_incumbent_comparison_and_feedback(self):
        optimizer = create_optimizer(self.settings(), [0.5])
        self.assertIsNone(optimizer.best())
        plus = optimizer.ask()[0].unit
        feedback = optimizer.tell({"plus": [100], "minus": [0], "incumbent": [99]})
        self.assertEqual(feedback.best_id, "plus")
        self.assertIsNone(feedback.best_latest)
        self.assertEqual(optimizer.best().unit, plus)
        feedback = optimizer.tell({"plus": [2], "minus": [1], "incumbent": [3]})
        self.assertEqual(feedback.best_id, "incumbent")
        self.assertEqual(feedback.best_latest, 3)
        self.assertEqual(optimizer.best().mean, 51.5)
        self.assertEqual(optimizer.best().evaluations, 2)
        minus = optimizer.ask()[1].unit
        feedback = optimizer.tell({"plus": [2], "minus": [4], "incumbent": [3]})
        self.assertEqual(feedback.best_id, "minus")
        self.assertEqual(optimizer.best().unit, minus)
        self.assertEqual(optimizer.best().evaluations, 1)
        self.assertEqual(optimizer.evaluation_counts(0), [])
        self.assertEqual(optimizer.evaluation_counts(3), [3, 3, 3])

    def test_validation_before_advancing(self):
        for center in ([], [float("nan")], [-0.1], [1.1], [[0.5]]):
            with self.assertRaises(ValueError):
                create_optimizer(self.settings(), center)
        for key in optimizer_defaults("spsa"):
            for value in (-1, float("nan"), float("inf")):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    create_optimizer(self.settings(**{key: value}), [0.5])
        for key in ("spsa_learning_rate", "spsa_perturbation", "spsa_alpha", "spsa_gamma", "spsa_max_step"):
            with self.assertRaises(ValueError):
                create_optimizer(self.settings(**{key: 0}), [0.5])
        for key in ("spsa_perturbation", "spsa_start_jitter"):
            with self.assertRaises(ValueError):
                create_optimizer(self.settings(**{key: 0.6}), [0.5])
        optimizer = create_optimizer(self.settings(), [0.5])
        saved = json.loads(json.dumps(optimizer.state()))
        for scores in ({}, {"plus": [], "minus": [1], "incumbent": [1]},
                       {"plus": [float("nan")], "minus": [1], "incumbent": [1]},
                       {"plus": [1, 2], "minus": [1], "incumbent": [1]}):
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
                    weight = json.load(f)["weight"]
                count = int(cmd[cmd.index("--games") + 1])
                game = {"rules": "tetrio", "attack": 100 * weight, "died": False,
                        "sims_per_move": 128, "ms_per_move": 1 / (1 + weight)}
                return 0, "\n".join(json.dumps(game) for _ in range(count)), ""

            with patch.object(Runner, "print_config", return_value={"weight": 0.2}), \
                    patch.object(Runner, "_spawn", new=spawn), patch.object(Run, "health_checks"):
                for objective in ("strength", "sims-per-second"):
                    run = Run.create(os.path.join(directory, objective), binary, [0], [space],
                                     {"optimizer": "spsa", "objective": objective, "sims": 128,
                                      "iterations": 4, "games_per_eval": 2, "gate_games": 8,
                                      "spsa_learning_rate": 0.02})
                    self.assertEqual(run.settings["spsa_learning_rate"], 0.02)
                    self.assertNotIn("particles", run.settings)
                    run.run(plan_only=True)
                    run.baseline()
                    run.iterate()
                    run.save()
                    resumed = Run(run.dir)
                    self.assertEqual(run.optimizer.state(), resumed.optimizer.state())
                    resumed.run()
                    final = Run(run.dir)
                    self.assertEqual(final.state["phase"], "done")
                    self.assertEqual(final.optimizer.name, "spsa")
                    self.assertEqual(final.optimizer.iteration, 4)
                    with open(final.path("evals.jsonl")) as f:
                        records = [json.loads(line) for line in f]
                    self.assertEqual(len(records), 4)
                    self.assertTrue(all(r["optimizer"] == "spsa" and len(r["scores"]) == 3 for r in records))
                    self.assertEqual(len(final.state["holdouts"]), 1)


if __name__ == "__main__":
    unittest.main()
