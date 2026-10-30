import math
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from .gate import run_gate, score_sprt
from .run import Run
from .runner import RunnerError, Stage
from .stadium import StadiumRunner


class StadiumGateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.runner = StadiumRunner("unused", [0, 2], directory.name, "stadium", "bot")
        self.baseline = {"version": "tuning baseline"}
        self.release = {"version": "promoted release"}
        self.candidate = {"version": "candidate"}
        self.runner.baseline_config = self.baseline
        self.stage = Stage({"name": "gate", "objective": "stadium", "sims": 1024})

    def gate(self, score, **options):
        self.requests = []

        def play(requests):
            self.assertEqual(len(requests), 1)
            key, config, _, seed, n = requests[0]
            self.assertEqual(key, "cand")
            self.assertIs(config, self.candidate)
            self.assertIs(self.runner.baseline_config, self.release)
            self.requests.append((seed, n))
            return {key: [{"match_score": score}] * n}

        with patch.object(self.runner, "play", side_effect=play):
            result = run_gate(self.runner, self.stage, self.candidate, self.release, 100,
                              log=lambda _: None, **options)
        self.assertIs(self.runner.baseline_config, self.baseline)
        self.assertEqual(result["games"], sum(n for _, n in self.requests))
        self.assertEqual(result["seed_end"], 100 + result["games"])
        return result

    def test_strong_candidate_accepts_without_release_self_play(self):
        result = self.gate(1.0, max_games=256, batch=16)
        self.assertEqual(result["verdict"], "accept")
        self.assertLess(result["games"], 256)
        self.assertEqual(result["wins"], result["games"])
        self.assertEqual(result["draws"], 0)
        self.assertEqual(result["losses"], 0)
        self.assertEqual(result["null_score"], 0.5)
        self.assertEqual(result["alternative_score"], 0.51)
        self.assertEqual(result["opponent"], "release")

    def test_weak_candidate_rejects(self):
        result = self.gate(0.0, max_games=256, batch=16)
        self.assertEqual(result["verdict"], "reject")
        self.assertEqual(result["losses"], result["games"])

    def test_short_unanimous_sample_is_inconclusive(self):
        result = self.gate(1.0, max_games=8)
        self.assertEqual(result["verdict"], "inconclusive")
        self.assertTrue(0 < result["llr"] < 1)

    def test_draws_are_half_wins_and_partial_batch_respects_cap(self):
        result = self.gate(0.5, max_games=10, batch=6)
        self.assertEqual(self.requests, [(100, 6), (106, 4)])
        self.assertEqual(result["verdict"], "inconclusive")
        self.assertEqual(result["draws"], 10)
        self.assertEqual(result["candidate"], 0.5)
        self.assertAlmostEqual(result["llr"], 5 * math.log(1.02 * 0.98))

    def test_restores_tuning_opponent_on_failure_and_interruption(self):
        for error in (RunnerError("broken match"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                with patch.object(self.runner, "scores", side_effect=error):
                    with self.assertRaises(type(error)):
                        run_gate(self.runner, self.stage, self.candidate, self.release, 100)
                self.assertIs(self.runner.baseline_config, self.baseline)

    def test_score_likelihood_matches_known_bernoulli_case(self):
        llr, _ = score_sprt([1, 0, 0.5], 0.75, 0.05, 0.05)
        self.assertAlmostEqual(llr, 1.5 * math.log(0.75))

    def test_projection_removes_only_the_extra_stadium_games(self):
        run = object.__new__(Run)
        run.settings = {"objective": "strength", "chunk": 4, "games_per_eval": 32,
                        "iterations": 2, "holdout": {"every": 5, "games": 32},
                        "gate": {"max_games": 400}}
        run.runner = SimpleNamespace(workers=2)
        run.optimizer = Mock()
        run.optimizer.evaluation_counts.return_value = [1, 1]
        paired = run.projection(2)["gate"]
        run.settings["objective"] = "stadium"
        direct = run.projection(2)["gate"]
        self.assertAlmostEqual(paired - direct, 400 / 3600)


if __name__ == "__main__":
    unittest.main()
