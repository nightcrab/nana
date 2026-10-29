import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from .gate import run_gate
from .run import Run
from .runner import Runner, RunnerError, Stage
from .stadium import StadiumRunner


class StadiumTests(unittest.TestCase):
    def test_matches_isolate_configs_alternate_sides_and_validate_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bot = root / "nana_tbp.exe"
            bot.write_bytes(b"test executable")
            original = root / "nana.json"
            original.write_text('{"height": 99}')
            runner = StadiumRunner("unused", [0, 2, 4, 6], directory, "stadium", bot, chunk=1)
            runner.baseline_config = {"height": 1}
            self.assertEqual(runner.workers, 2)
            stage = Stage({"name": "versus", "sims": 128, "objective": "stadium", "rules": "botris"})
            calls = []

            def spawn(cmd, env, cwd, cpus):
                work = Path(cwd)
                settings = json.loads((work / "stadium_cli_settings.json").read_text())
                self.assertEqual(settings["thinks"], 128)
                self.assertEqual(settings["garbage_delay"], 1)
                self.assertEqual(len(cpus), 2)
                configs = [json.loads(Path(path).with_name("nana.json").read_text()) for path in cmd[1:]]
                self.assertEqual(sorted(c["height"] for c in configs), [1, 2])
                self.assertTrue(all(c["rules"] == "botris" and c["threads"] == 1 for c in configs))
                side = next(i + 1 for i, c in enumerate(configs) if c["height"] == 2)
                calls.append(side)
                with closing(sqlite3.connect(settings["db_path"])) as db, db:
                    db.execute("CREATE TABLE Data (game_id INTEGER, state TEXT)")
                    db.execute("INSERT INTO Data VALUES (1, 'PLAYING')")
                    db.execute("INSERT INTO Data VALUES (1, ?)", (f"P{side}_WIN",))
                return 0, "", ""

            with patch.object(runner, "_spawn", side_effect=spawn):
                scores, _ = runner.scores([("candidate", {"height": 2}, stage, 100, 3)])
            self.assertEqual(scores["candidate"], [1, 1, 1])
            self.assertEqual(sorted(calls), [1, 1, 2])
            self.assertEqual(runner.games_played, 3)
            self.assertEqual(original.read_text(), '{"height": 99}')
            self.assertEqual(list(Path(runner.match_dir).iterdir()), [])

            with patch.object(runner, "_spawn", return_value=(1, "board diagnostics", "bot crashed")):
                with self.assertRaisesRegex(RunnerError, "bot crashed"):
                    runner.play([("bad", {"height": 2}, stage, 100, 1)])
            self.assertEqual(runner.free.qsize(), 2)
            failures = list((root / "failed-matches").iterdir())
            self.assertEqual(len(failures), 1)
            failure = failures[0]
            self.assertEqual((failure / "stdout.log").read_text(), "board diagnostics")
            self.assertEqual((failure / "stderr.log").read_text(), "bot crashed")
            self.assertEqual(json.loads((failure / "match.json").read_text())["candidate_side"], 1)
            self.assertEqual(json.loads((failure / "candidate" / "nana.json").read_text())["height"], 2)

    def test_run_persists_fixed_baseline_and_gate_uses_match_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "binary"
            binary.touch()
            space = root / "space.json"
            space.write_text(json.dumps([{"name": "height", "default": 1, "low": 0, "high": 2}]))
            options = {"objective": "stadium", "sims": 128, "stadium": str(binary), "bot": str(binary)}
            with patch.object(Runner, "print_config", return_value={"height": 1}):
                run = Run.create(str(root / "run"), str(binary), [0, 2], [str(space)], options)
            resumed = Run(run.dir)
            self.assertIsInstance(resumed.runner, StadiumRunner)
            self.assertEqual(resumed.runner.baseline_config, {"height": 1})
            resumed.state["plan"] = {"sims": 128}
            stage = resumed.stage()
            self.assertEqual(stage.objective, "stadium")

            def play(requests):
                return {key: [{"match_score": config["score"]} for _ in range(n)]
                        for key, config, _, _, n in requests}

            with patch.object(resumed.runner, "play", side_effect=play):
                result = run_gate(resumed.runner, stage, {"score": 1}, {"score": 0.5}, 100,
                                  max_games=8, log=lambda _: None)
            self.assertEqual(result["verdict"], "accept")
            self.assertEqual(result["candidate"], 1)
            self.assertEqual(result["release"], 0.5)

    def test_requires_cpus_for_both_bots(self):
        for cpus, threads in (([0], 1), ([0, 1, 2, 3, 4, 5], 2)):
            with self.assertRaisesRegex(ValueError, "two CPU groups"):
                StadiumRunner.validate_cpus(cpus, threads)


if __name__ == "__main__":
    unittest.main()
