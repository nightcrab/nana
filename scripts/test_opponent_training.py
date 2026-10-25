"""Checks for the offline experiment's leakage boundaries and distributional loss."""

import unittest
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

import numpy as np
import torch

from opponent_model import Predictor
from opponent_data import positions_from_db, target_player
from train_opponent import sequence_loss, split_games, gain_intervals


class TrainingTests(unittest.TestCase):
    def test_target_side_follows_metadata_and_rejects_ambiguous_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"games.db"
            sides = Path(tmp)/"sides.txt"
            sides.write_text("p1 cc-tbp p2 tbp_tetris")
            self.assertEqual(target_player(path, "auto"), 1)
            sides.write_text("p1 tbp_tetris p2 cc-tbp")
            self.assertEqual(target_player(path, "auto"), 2)
            self.assertEqual(target_player(path, "1"), 1)
            sides.write_text("p1 cc-tbp p2 cc-tbp")
            with self.assertRaises(ValueError):
                target_player(path, "auto")

    def test_stadium_cancellation_and_end_of_game_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"stadium.db"
            fields = ["game_id INTEGER", "move_index INTEGER", "state TEXT"]
            for p in (1, 2):
                fields += [f"p{p}_attack INTEGER", f"p{p}_meter INTEGER"]
            fields += ["p2_board BLOB", "p2_current_piece TEXT", "p2_hold TEXT",
                       "p2_combo INTEGER", "p2_b2b INTEGER"]
            fields += [f"p2_queue_{i} TEXT" for i in range(5)]
            with closing(sqlite3.connect(path)) as db:
                db.execute("CREATE TABLE Data (" + ",".join(fields) + ")")
                pieces = "IOTLJSZ" * 4
                for gid, terminal in ((1, "P1_WIN"), (2, "P2_WIN")):
                    for step in range(8):
                        values = [gid, step, "PLAYING" if step < 7 else terminal,
                                  5, 2, 0, 0, bytes(200), pieces[step], "NULL", 0, 0]
                        values += list(pieces[step+1:step+6])
                        db.execute("INSERT INTO Data VALUES (" + ",".join("?" for _ in values) + ")", values)
                db.commit()
            positions = positions_from_db(path, player=2, horizon=8, stride=1)
            loss = next(p for p in positions if p["game"] == "1" and p["move"] == 0)
            win = next(p for p in positions if p["game"] == "2" and p["move"] == 0)
            self.assertEqual(loss["incoming"], [3]*7)
            self.assertEqual(loss["recorded"][-1], 8)
            self.assertTrue(loss["real_mask"][-1])
            self.assertFalse(win["real_mask"][-1])
            self.assertFalse(loss["replay_mask"][-1])

    def test_split_keeps_all_windows_of_game_together(self):
        games = np.repeat([f"game-{i}" for i in range(20)], 5)
        splits = split_games(games, 17)
        sets = [set(games[indices]) for indices in splits.values()]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(sum(map(len, splits.values())), len(games))

    def test_replaying_same_samples_does_not_change_weight(self):
        logits = torch.randn(2, 3, 9)
        real = torch.zeros(2, 3, dtype=torch.long)
        mask = torch.ones(2, 3, dtype=torch.bool)
        replay = torch.ones(2, 2, 3, dtype=torch.long)
        first = sequence_loss(logits, real, mask, replay, mask, 0.5)
        second = sequence_loss(logits, real, mask, replay.repeat(1, 4, 1), mask, 0.5)
        torch.testing.assert_close(first, second)

    def test_censored_targets_do_not_contribute(self):
        logits = torch.zeros(1, 3, 9, requires_grad=True)
        real = torch.zeros(1, 3, dtype=torch.long)
        mask = torch.tensor([[True, False, False]])
        replay = real[:, None]
        loss = sequence_loss(logits, real, mask, replay, mask, 0.5)
        loss.backward()
        self.assertEqual(float(logits.grad[:, 1:].abs().sum()), 0)
        self.assertGreater(float(logits.grad[:, :1].abs().sum()), 0)

    def test_model_uses_raw_cells_and_outputs_normalized_distributions(self):
        torch.manual_seed(2)
        torch.set_num_threads(1)
        model = Predictor(width=8, blocks=1)
        boards = torch.zeros(2, 1, 20, 10)
        boards[1, 0, :5, :5] = 1
        output = model(boards, torch.zeros(2, 3))
        self.assertEqual(tuple(output.shape), (2, 8, 9))
        self.assertFalse(torch.allclose(output[0], output[1]))
        torch.testing.assert_close(output.softmax(-1).sum(-1), torch.ones(2, 8))

    def test_bootstrap_identical_predictions_have_zero_gain(self):
        prob = torch.ones(4, 8, 9)/9
        target = torch.zeros(4, 8, dtype=torch.long)
        result = gain_intervals(prob, prob, target, torch.ones(4, 8, dtype=torch.bool),
                                np.array(["a", "a", "b", "b"]), 1)
        self.assertEqual(result["8"], [0, 0])


if __name__ == "__main__":
    unittest.main()
