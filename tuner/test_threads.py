import json
import os
import tempfile
import unittest
from unittest.mock import patch

from .run import Run
from .runner import Runner, RunnerError, Stage


class ThreadTests(unittest.TestCase):
    def test_evaluator_is_loaded_and_preserved_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = os.path.join(directory, "nana")
            with open(binary, "w"):
                pass
            space = os.path.join(directory, "space.json")
            with open(space, "w") as f:
                json.dump([{"name": "steal_topk", "default": 4, "low": 1, "high": 8}], f)
            for mode in ("cc", "convex"):
                with self.subTest(mode=mode), patch.object(Runner, "print_config",
                        side_effect=lambda config: dict(config)) as loaded:
                    run = Run.create(os.path.join(directory, mode), binary, [0], [space],
                                     {"sims": 128, "eval": mode})
                    loaded.assert_called_once_with({"eval": mode})
                resumed = Run(run.dir)
                self.assertEqual(resumed.state["base_config"]["eval"], mode)
                self.assertEqual(resumed.config(resumed.space.default_unit())["eval"], mode)
                with open(run.path("release.json")) as f:
                    self.assertEqual(json.load(f)["eval"], mode)

    def test_invalid_allocations_are_rejected(self):
        for cpus, threads in (([], 1), ([0, 0], 1), ([-1], 1), ([0], 0),
                              ([0], 65), ([0], 2), ([0, 2, 4], 2)):
            with self.subTest(cpus=cpus, threads=threads), self.assertRaises(ValueError):
                Runner.validate_cpus(cpus, threads)

    def test_games_receive_exclusive_cpu_groups_and_requested_threads(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = Runner("unused", [0, 2, 4, 6], directory, threads=2, chunk=1)
            stage = Stage({"name": "test", "sims": 128})
            commands = []

            def spawn(cmd, env):
                commands.append(cmd)
                return 0, json.dumps({"rules": "tetrio", "attack": 1}), ""

            with patch.object(runner, "_spawn", side_effect=spawn):
                results = runner.play([("test", {}, stage, 1, 4)])
            self.assertEqual(len(results["test"]), 4)
            self.assertEqual(runner.workers, 2)
            for cmd in commands:
                self.assertEqual(cmd[cmd.index("--threads") + 1], "2")
                self.assertIn(cmd[cmd.index("--cpus") + 1], ("0,2", "4,6"))
                self.assertNotIn("--cpu", cmd)
            self.assertEqual(runner.free.qsize(), 2)

    def test_failed_game_returns_its_cpu_group(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = Runner("unused", [0, 2], directory, threads=2)
            with patch.object(runner, "_spawn", return_value=(1, "", "failed")):
                with self.assertRaises(RunnerError):
                    runner._run_chunk("unused.json", Stage({"name": "test", "sims": 128}), 1, 1)
            self.assertEqual(runner.free.get_nowait(), [0, 2])

    def test_resume_preserves_threads_and_health_checks_allow_scheduling_noise(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = os.path.join(directory, "nana")
            with open(binary, "w"):
                pass
            space = os.path.join(directory, "space.json")
            with open(space, "w") as f:
                json.dump([{"name": "steal_topk", "default": 4, "low": 1, "high": 8,
                            "integer": True}], f)
            with patch.object(Runner, "print_config", return_value={"steal_topk": 4}):
                run = Run.create(os.path.join(directory, "run"), binary, [0, 2, 4, 6], [space],
                                 {"threads": 2, "sims": 128, "objective": "sims-per-second"})
            resumed = Run(run.dir)
            self.assertEqual(resumed.runner.threads, 2)
            self.assertEqual(resumed.runner.workers, 2)
            with patch("tuner.run.check_plumbing") as plumbing, \
                    patch("tuner.run.check_determinism") as determinism, \
                    patch.object(resumed.runner, "scores") as scores:
                resumed.health_checks()
                plumbing.assert_called_once()
                determinism.assert_not_called()
                scores.assert_called_once()

            # Existing runs without the new setting retain their original allocation.
            settings = resumed.settings.copy()
            del settings["threads"]
            with open(run.path("run.json"), "w") as f:
                json.dump(settings, f)
            legacy = Run(run.dir)
            self.assertEqual(legacy.runner.threads, 1)
            self.assertEqual(legacy.runner.workers, 4)
            with patch("tuner.run.check_plumbing"), \
                    patch("tuner.run.check_determinism") as determinism, \
                    patch.object(legacy.runner, "scores"):
                legacy.health_checks()
                determinism.assert_called_once()


if __name__ == "__main__":
    unittest.main()
