"""Checks for the offline experiment's leakage boundaries and distributional loss."""

import unittest
import itertools
import sqlite3
import tempfile
import json
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import closing
from pathlib import Path

import numpy as np
import torch

from opponent_model import Predictor
from opponent_data import positions_from_db, target_player, generate
from train_opponent import sequence_loss, split_games, gain_intervals, trajectory_metrics


class TrainingTests(unittest.TestCase):
    def test_no_garbage_uses_only_generated_full_horizon_targets_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'games.db'
            path.write_bytes(b'dummy database fingerprint')
            exe = Path(tmp)/'replay'
            exe.write_bytes(b'dummy executable fingerprint')
            position = dict(game='1', board=[[0]*10 for _ in range(40)], queue=list('IOTLJS'),
                            hold=None, bag=['Z'], combo=0, b2b=0, meter=7,
                            incoming=[9], recorded=[6, 6, 6], real_mask=[True, False, False],
                            replay_mask=[True, False, False])
            args = SimpleNamespace(dbs=[str(path)], player='2', horizon=3, stride=1,
                                   max_positions=0, seed=1, samples=2, thinks=4000,
                                   replay_weight=0.5, cc=str(exe), jobs=1, shard_size=32,
                                   cache=str(Path(tmp)/'cache'), pieces=True, no_garbage=True,
                                   prepare_only=False, exclude_positions=[])
            rollouts = [dict(position=0, queue_seed=q, repeat=0, moves=[dict(attack=q)]) for q in range(2)]
            with patch('opponent_data.positions_from_db', return_value=[position]), \
                 patch('opponent_data.cc.run_replays', return_value=rollouts) as replay:
                arrays, manifest = generate(args)
                np.testing.assert_array_equal(arrays['real'], [[0, 8, 8], [1, 8, 8]])
                self.assertTrue(arrays['real_mask'].all())
                self.assertEqual(arrays['replay'].shape, (2, 0, 3))
                self.assertEqual(len(set(arrays['game'])), 1)
                self.assertEqual(manifest[0]['target_mode'], 'no-garbage-continuations')
                args.prepare_only = True
                prepared, summary = generate(args)
                self.assertEqual(prepared, {})
                self.assertEqual(summary[0]['examples'], 2)
                self.assertEqual(replay.call_count, 1)
                # Excluding an evaluation root changes the cache and removes it.
                excluded = Path(tmp)/'held-out.jsonl'
                excluded.write_text(json.dumps(dict(positions=[position]))+'\n')
                args.exclude_positions = [str(excluded)]
                _, summary = generate(args)
                self.assertEqual(summary[0]['examples'], 0)
                self.assertEqual(replay.call_count, 1)

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
        replay_logits = torch.randn(2, 2, 3, 9)
        first = sequence_loss(logits, real, mask, replay, mask, 0.5, replay_logits)
        second = sequence_loss(logits, real, mask, replay.repeat(1, 4, 1), mask, 0.5,
                               replay_logits.repeat(1, 4, 1, 1))
        torch.testing.assert_close(first, second)

    def test_censored_targets_do_not_contribute(self):
        logits = torch.zeros(1, 3, 9, requires_grad=True)
        real = torch.zeros(1, 3, dtype=torch.long)
        mask = torch.tensor([[True, False, False]])
        replay = real[:, None]
        loss = sequence_loss(logits, real, mask, replay, mask, 0.5, logits[:, None])
        loss.backward()
        self.assertEqual(float(logits.grad[:, 1:].abs().sum()), 0)
        self.assertGreater(float(logits.grad[:, :1].abs().sum()), 0)

    def test_model_uses_raw_cells_and_outputs_normalized_distributions(self):
        torch.manual_seed(2)
        torch.set_num_threads(1)
        model = Predictor(width=2, blocks=1, heads=32, hidden=8, fusion=16)
        boards = torch.zeros(2, 1, 20, 10)
        boards[1, 0, :5, :5] = 1
        output = model(boards, torch.zeros(2, 3), torch.zeros(2, 8, dtype=torch.long))
        self.assertEqual(tuple(output.shape), (2, 8, 9))
        self.assertFalse(torch.allclose(output[0], output[1]))
        torch.testing.assert_close(output.softmax(-1).sum(-1), torch.ones(2, 8))
        output.sum().backward()
        self.assertEqual(len(model.heads), 32)
        for head in model.heads:
            self.assertIsNotNone(head[0].weight.grad)

    def test_teacher_forcing_is_causal_and_uses_each_replay_prefix(self):
        torch.manual_seed(3)
        model = Predictor(horizon=3, board=False, hidden=8, fusion=8)
        board, context = torch.zeros(2, 1, 20, 10), torch.zeros(2, 3)
        targets = torch.tensor([[0, 0, 0], [1, 0, 2]])
        output = model(board, context, targets)
        torch.testing.assert_close(output[0, 0], output[1, 0])
        self.assertFalse(torch.allclose(output[0, 1], output[1, 1]))
        changed = targets.clone()
        changed[:, -1] = 5
        torch.testing.assert_close(output, model(board, context, changed))

    def test_replay_training_matches_individual_sequence_losses(self):
        torch.manual_seed(5)
        model = Predictor(horizon=3, board=False, hidden=8, fusion=8)
        conditioning = model.encode(torch.zeros(1, 1, 20, 10), torch.zeros(1, 3))
        real = torch.tensor([[0, 1, 0]])
        replay = torch.tensor([[[1, 0, 2], [8, 8, 8]]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        real_logits = model.decode(conditioning, real)
        replay_logits = model.decode(conditioning.expand(2, -1), replay[0])[None]
        combined = sequence_loss(real_logits, real, mask, replay, mask, 0.5, replay_logits)
        expected = -real_logits.log_softmax(-1).gather(-1, real[..., None]).mean()
        for sample in replay[0]:
            logits = model.decode(conditioning, sample[None])
            expected = expected - 0.25 * logits.log_softmax(-1).gather(-1, sample[None, :, None]).mean()
        torch.testing.assert_close(combined, expected)
        combined.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_beam_matches_exhaustive_joint_probabilities_and_death_is_absorbing(self):
        torch.manual_seed(4)
        model = Predictor(horizon=2, board=False, hidden=8, fusion=8)
        model.eval()
        targets = torch.tensor(list(itertools.product(range(9), repeat=2)))
        board, context = torch.zeros(81, 1, 20, 10), torch.zeros(81, 3)
        logits = model(board, context, targets)
        scores = logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1).sum(-1)
        self.assertAlmostEqual(float(scores.exp().sum().detach()), 1, places=5)
        expected, indices = scores.topk(3)
        sequences, actual = model.trajectories(board[:2], context[:2], beam=81)
        for b in range(2):
            torch.testing.assert_close(actual[b], expected)
            torch.testing.assert_close(sequences[b], targets[indices])
        self.assertTrue(torch.isneginf(scores[(targets[:, 0] == 8) & (targets[:, 1] != 8)]).all())

    def test_coverage_excludes_censored_windows_and_preserves_absolute_mass(self):
        sequences = torch.tensor([[[0, 0], [1, 0], [2, 0]], [[0, 0], [1, 0], [2, 0]]])
        logp = torch.tensor([[0.4, 0.2, 0.1], [0.4, 0.2, 0.1]]).log()
        metrics = trajectory_metrics(sequences, logp, torch.tensor([[1, 0], [0, 0]]),
                                     torch.tensor([[True, True], [True, False]]))
        self.assertEqual(metrics["complete_windows"], 1)
        self.assertEqual(metrics["top1_exact_coverage"], 0)
        self.assertEqual(metrics["top3_exact_coverage"], 1)
        self.assertAlmostEqual(metrics["mean_top3_probability_mass"], 0.7, places=6)

    def test_bootstrap_identical_predictions_have_zero_gain(self):
        prob = torch.ones(4, 8, 9)/9
        target = torch.zeros(4, 8, dtype=torch.long)
        result = gain_intervals(prob, prob, target, torch.ones(4, 8, dtype=torch.bool),
                                np.array(["a", "a", "b", "b"]), 1)
        self.assertEqual(result["8"], [0, 0])


if __name__ == "__main__":
    unittest.main()
