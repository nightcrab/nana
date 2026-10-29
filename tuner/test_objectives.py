import json
import os
import tempfile
import unittest
from unittest.mock import patch

from .gate import run_gate
from .run import Run
from .runner import Runner, RunnerError, Stage, fitness


class ObjectiveTests(unittest.TestCase):
    def test_fitness_uses_search_throughput_not_game_length_or_attack(self):
        game = {"attack": 200, "pieces": 100, "sims_per_move": 1000, "ms_per_move": 2}
        self.assertEqual(fitness(game, 100), 2)
        self.assertEqual(fitness(game, 100, "sims-per-second"), 500_000)
        game.update(attack=0, pieces=1)
        self.assertEqual(fitness(game, 100, "sims-per-second"), 500_000)

    def test_game_score_fitness_uses_emulator_metric(self):
        self.assertEqual(fitness({"game_score": 17}, 100, "game-score"), 17)
        with self.assertRaises(RunnerError):
            fitness({"lines": 17}, 100, "game-score")

    def test_gate_accepts_game_score_improvement(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = Runner("unused", [0], directory)
            stage = Stage({"name": "score", "sims": 1024, "objective": "game-score"})

            def play(requests):
                return {key: [{"game_score": config["game_score"]} for _ in range(n)]
                        for key, config, _, _, n in requests}

            with patch.object(runner, "play", side_effect=play):
                result = run_gate(runner, stage, {"game_score": 3}, {"game_score": 2},
                                  100, max_games=8, log=lambda _: None)
            self.assertEqual(result["verdict"], "accept")
            self.assertEqual(result["candidate"], 3)

    def test_bad_timing_is_rejected(self):
        for ms in (0, -1, float("nan"), float("inf")):
            with self.subTest(ms=ms), self.assertRaises(RunnerError):
                fitness({"sims_per_move": 1000, "ms_per_move": ms}, 100, "sims-per-second")
        with self.assertRaises(RunnerError):
            fitness({"attack": 1}, 100, "sims-per-second")

    def test_gate_accepts_faster_candidate_with_lower_attack(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = Runner("unused", [0], directory)
            stage = Stage({"name": "speed", "sims": 1024, "objective": "sims-per-second"})

            def play(requests):
                return {key: [{"attack": config["attack"], "sims_per_move": 1024,
                               "ms_per_move": config["ms"]} for _ in range(n)]
                        for key, config, _, _, n in requests}

            with patch.object(runner, "play", side_effect=play):
                result = run_gate(runner, stage, {"attack": 0, "ms": 1}, {"attack": 100, "ms": 2},
                                  100, max_games=8, log=lambda _: None)
            self.assertEqual(result["verdict"], "accept")
            self.assertEqual(result["candidate"], 1_024_000)

    def test_speed_run_checkpoint_and_single_game_evaluations(self):
        # Exercise calibration, PSO, holdouts, gate, and resume without a C++ binary.
        with tempfile.TemporaryDirectory() as directory:
            binary = os.path.join(directory, "nana")
            with open(binary, "w"):
                pass
            space = os.path.join(directory, "space.json")
            with open(space, "w") as f:
                json.dump([{"name": "speed", "default": 1, "low": 0.5, "high": 2}], f)

            def spawn(runner, cmd, env):
                with open(env["NANA_CONFIG"]) as f:
                    config = json.load(f)
                n = int(cmd[cmd.index("--games") + 1])
                game = {"rules": "tetrio", "attack": 0, "died": False,
                        "sims_per_move": 1024, "ms_per_move": 2 / config.get("speed", 1)}
                return 0, "\n".join(json.dumps(game) for _ in range(n)), ""

            with patch.object(Runner, "print_config", return_value={"speed": 1}), \
                    patch.object(Runner, "_spawn", new=spawn), patch.object(Run, "health_checks"):
                for objective, expected in (("strength", (32, 32, 400)),
                                            ("sims-per-second", (4, 8, 32)),
                                            ("game-score", (32, 32, 400))):
                    defaults = Run.create(os.path.join(directory, objective), binary, [0], [space],
                                          {"objective": objective, "sims": 1024})
                    self.assertEqual((defaults.settings["games_per_eval"], defaults.settings["holdout"]["games"],
                                      defaults.settings["gate"]["max_games"]), expected)
                run = Run.create(os.path.join(directory, "run"), binary, [0], [space],
                                 {"objective": "sims-per-second", "sims": 1024, "particles": 2,
                                  "iterations": 2, "games_per_eval": 1, "gate_games": 8})
                self.assertEqual(run.settings["holdout"]["games"], 8)
                self.assertEqual(run.settings["games_per_eval"], 1)
                self.assertEqual(run.settings["gate"]["max_games"], 8)
                run.run()
                resumed = Run(run.dir)
                self.assertEqual(resumed.state["phase"], "done")
                self.assertEqual(resumed.stage().objective, "sims-per-second")
                self.assertEqual(resumed.state["baseline"]["mean"], 512_000)
                self.assertEqual(len(resumed.state["holdouts"]), 1)
                with open(run.path("evals.jsonl")) as f:
                    records = [json.loads(line) for line in f]
                self.assertTrue(all(r["objective"] == "sims-per-second" for r in records))
                # Old checkpoints have no objective and must keep strength scoring.
                settings = resumed.settings.copy()
                del settings["objective"]
                with open(run.path("run.json"), "w") as f:
                    json.dump(settings, f)
                self.assertEqual(Run(run.dir).stage().objective, "strength")


if __name__ == "__main__":
    unittest.main()
