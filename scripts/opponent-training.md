# Offline board prediction experiment

This is a CPU-only experiment: raw stadium boards -> residual CNN -> eight categorical attack
distributions. The output is a deterministic latent chain, with conditionally independent emissions.
There is no C++ inference, bot integration, or dependency on eval features or fitted HMM states.

Requirements: Python with NumPy and CPU PyTorch; Rust/Cargo; a sibling `../cold-clear` checkout
with the TETR.IO damage changes used by the existing entropy experiment. The replay helper's Cargo
manifest uses that sibling checkout directly. It does not change its files.

From the nana root, run (PowerShell or a shell; write paths explicitly):

```
python scripts/train_opponent.py results/stadium/games.db --player 2 --out results/opponent-cnn
```

This builds the standalone replay helper, extracts positions, generates/cache-shards CC continuations,
trains a context-only baseline and CNN, selects checkpoints using validation games, and reports
performance on separate test games. At least three games are required; meaningful evaluation needs
many more. Multiple database paths are supported. The selected player should be the intended target
bot; the script cannot discover bot identity from the database.

The existing EC2 collection is already copied to `ec2-results/h2h-20261025/stadium_*/games.db`:
678 nana-vs-Cold-Clear games, with Cold Clear on P2 in stadiums 0/1 and P1 in 2/3.
Use `--player auto` to read the adjacent `sides.txt` metadata and select Cold Clear correctly.
In PowerShell, a first real-only experiment on this collection is:

```powershell
$gameDbs = @(Get-ChildItem ec2-results/h2h-20261025/stadium_*/games.db | ForEach-Object FullName)
python scripts/train_opponent.py @gameDbs --player auto --replay-weight 0 --width 16 --out results/cc-board-real
```

Remove `--replay-weight 0` and choose another output directory to include CC replay supervision.

Default encoder: occupancy and two coordinate channels; 32-channel stem; four residual blocks
(two 3x3 convolutions each); 8-channel 1x1 projection; flatten; 256-unit dense head. No pooling.
Context is log1p(combo, B2B, meter). `--pieces` additionally supplies one-hot hold, current piece,
and five preview pieces; enable only if those are available to the intended predictor.
The context-only baseline receives exactly the same context. The pooled baseline is a smoothed,
per-horizon categorical frequency distribution fitted only on training games.

Replay supervision samples legal bag continuations beyond the recorded preview. It fixes external
incoming attacks to the other player's recorded gross attack minus their recorded old meter,
floored at zero. These are the attacks after the other player's own cancellation and before
head-on cancellation. The replayed player's cancellation, garbage receipt, board, and attacks
evolve from its sampled actions. Garbage holes are sampled, with one hole per received batch.
Like the checked-in stadium implementation, zero-clear moves tank the whole meter; there is no
eight-line cap. Only zero garbage delay is supported: pending delayed attacks are not in this data.
This is a fixed-pressure approximation, not a replay of both players' decisions.

Recorded targets retain gross attack bins 0..6, 7+, and death, matching the entropy experiment.
After a recorded loss/draw, death is absorbing; after a recorded win or truncated game, unknown
targets are masked. Replay targets beyond the recorded pressure schedule are also masked.
The loss is recorded masked cross-entropy plus `--replay-weight` (default 0.5) times mean replay
cross-entropy. Repeats at a position are averaged, rather than increasing its weight.
No entropy penalty is used. Repeats share future queues but vary garbage holes and the bot's
internal search randomness; the CLI seed does not seed CC's internal search RNG.

Data is cached by database contents, loader source, replay executable, and generation settings.
Completed replay shards survive interruption. Changing model/training settings reuses generated
data. Each invocation trains anew; it does not resume optimizer state. To start without replay:

```
python scripts/train_opponent.py results/stadium/games.db --replay-weight 0 --out results/real-only
```

For a short pipeline check (this does not establish predictive power):

```
python scripts/train_opponent.py games.db --max-positions 128 --queue-seeds 2 --thinks 32 --width 16 --epochs 2
```

For a useful first experiment, increase distinct positions and search budget, and use e.g. eight
queue continuations with one repeat each. Datagen and training print progress separately. Tune
`--jobs` for replay and `--threads`/`--batch-size` for CPU training. `--width 16` offers a cheaper
encoder comparison. Defaults are starting settings, not calibrated runtime promises.

Artifacts in `--out`:

- `report.json`: settings, source/data fingerprints, exact game splits, epoch timing/loss,
  and untouched-test metrics at 1, 5, and 8 pieces (or the requested horizon).
- `cnn.pt`, `context.pt`: best validation checkpoints, including model settings.
- `test-predictions.npz`: distributions, real targets/masks, and game identifiers for analysis.

Positive `gain_vs_context` measures added information from board cells. Positive `gain_vs_pooled`
measures improvement over unconditional attack frequencies. Both are bits per piece; game-level
bootstrap intervals accompany them when at least two test games exist. Capped attack bias treats
7+ as 7 and death as zero; it does not measure uncapped attack bias. Attack Brier score measures
attack/no-attack calibration and discrimination together. Compare replay weights using validation
games; reserve test games for final assessment. Smoke runs on a few games are plumbing checks.

Checks:

```
python -m unittest discover -s scripts -p test_opponent_training.py
```

`opponent_replay/src/main.rs` is adapted from Cold Clear's `cc_replay.rs`, under Cold Clear's
MPL-2.0 license (included in `opponent_replay/LICENSE`). It adds fixed incoming attacks and garbage
application; the original entropy script is unchanged.
