import json
import os
import statistics
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .optimizer import Candidate, Feedback, Recommendation, create_optimizer, restore_optimizer
from .pso import PSO
from .run import DEFAULTS, Run
from .runner import Runner


class OptimizerTests(unittest.TestCase):
    def settings(self):
        from .optimizer import optimizer_defaults
        return json.loads(json.dumps({**DEFAULTS, **optimizer_defaults("pso"), "iterations": 6,
                                      "particles": 4, "seed": 7}))

    def test_adapter_preserves_original_pso_and_resume(self):
        settings = self.settings()
        center = [0.4, 0.6]
        reference = PSO(2, 4, rng_seed=7, inertia=0.9)
        reference.scatter(center, 0.3)
        optimizer = create_optimizer(settings, center)
        for iteration in range(6):
            points = reference.ask()
            candidates = optimizer.ask()
            self.assertEqual([p.id for p in candidates], [f"{k}{i}" for k, i, _ in points])
            for candidate, (_, _, unit) in zip(candidates, points):
                np.testing.assert_array_equal(candidate.unit, unit)
            scores = {c.id: [-sum((x - 0.7) ** 2 for x in c.unit) + offset
                             for offset in (-0.1, 0.1)] for c in candidates}
            reference.inertia = 0.9 + iteration / 5 * (0.4 - 0.9)
            reference.tell({(k, i): statistics.mean(scores[f"{k}{i}"]) for k, i, _ in points})
            feedback = optimizer.tell(scores)
            self.assertIn(feedback.best_id, scores)
            self.assertEqual(optimizer.state()["state"], reference.state())
            optimizer = restore_optimizer(settings, json.loads(json.dumps({"optimizer": optimizer.state()})))

    def test_legacy_run_migrates_and_continues_identically(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(directory)
            scores = {c.id: [0.5, 1.5] for c in run.optimizer.ask()}
            run.optimizer.tell(scores)
            run.state["iterations"] = 1
            run.save()
            expected = Run(run.dir)
            legacy_state = dict(run.state)
            legacy_state["pso"] = legacy_state.pop("optimizer")["state"]
            with open(run.path("state.json"), "w") as f:
                json.dump(legacy_state, f)
            settings = dict(run.settings)
            del settings["optimizer"]
            with open(run.path("run.json"), "w") as f:
                json.dump(settings, f)
            legacy = Run(run.dir)
            for optimizer in (expected.optimizer, legacy.optimizer):
                optimizer.tell({c.id: [1.0, 2.0] for c in optimizer.ask()})
            self.assertEqual(expected.optimizer.state(), legacy.optimizer.state())
            legacy.save()
            self.assertNotIn("pso", legacy.state)
            self.assertEqual(Run(run.dir).optimizer.state(), legacy.optimizer.state())

    def make_run(self, directory):
        binary = os.path.join(directory, "nana")
        with open(binary, "w"):
            pass
        space = os.path.join(directory, "space.json")
        with open(space, "w") as f:
            json.dump([{"name": "weight", "default": 0.5, "low": 0, "high": 1}], f)
        with patch.object(Runner, "print_config", return_value={"weight": 0.5}):
            return Run.create(os.path.join(directory, "run"), binary, [0], [space],
                              {"sims": 128, "particles": 2, "iterations": 3, "games_per_eval": 2})

    def test_controller_uses_opaque_ids_and_optimizer_cost_estimates(self):
        class OtherOptimizer:
            name = "other"

            def ask(self):
                return [Candidate("trial/arbitrary-id", [0.8])]

            def tell(self, scores):
                self.scores = scores
                return Feedback([statistics.mean(scores["trial/arbitrary-id"])], None, "trial/arbitrary-id")

            def best(self):
                return Recommendation("winner", [0.8], 1.5, 1)

            def evaluation_counts(self, iterations):
                return [1, 3, 5]

            def diagnostics(self):
                return {"radius": 0.2}

            def warnings(self, names):
                return ["custom: diagnostic"]

        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(directory)
            run.optimizer = OtherOptimizer()
            run.state["plan"] = {"sims": 128, "garbage_scale": 0.4,
                                 "projected_hours": {"iteration": 0.1, "gate": 0.1}}

            def evaluate(requests):
                self.assertEqual(len(requests), 1)
                key, config, _, seed, count = requests[0]
                self.assertEqual(config["weight"], 0.8)
                self.assertEqual(seed, 1_000_000)
                self.assertEqual(count, 2)
                return {key: [1.0, 2.0]}, {key: [{"died": False}] * 2}

            with patch.object(run.runner, "scores", side_effect=evaluate):
                best = run.iterate()
            self.assertEqual(best.index, "winner")
            self.assertIn("custom: diagnostic", run.warnings)
            run.status(best)
            self.assertEqual(run.projection(1)["tune"], (9 * 2 + 3 * 4) / 3600)
            self.assertEqual(run.projection(1)["iteration"], (5 * 2 + 4) / 3600)


if __name__ == "__main__":
    unittest.main()
