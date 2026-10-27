# Offline board prediction experiment

This is a CPU-only experiment: raw stadium boards -> 32 independent convolutional heads ->
concatenation and wide linear/ReLU fusion -> autoregressive GRU -> exact attack trajectories.
There is no C++ inference, bot integration, or dependency on eval features or fitted HMM states.

Requirements: Python with NumPy and CPU PyTorch; Rust/Cargo; a sibling `../cold-clear` checkout
with the TETR.IO damage changes used by the existing entropy experiment. The replay helper's Cargo
manifest uses that sibling checkout directly. It does not change its files.

## No-garbage continuation targets

`--no-garbage` takes real recorded root positions and uses generated Cold Clear
continuations as the **only** supervised targets, including validation and test.
No recorded game attacks contribute to the loss in this mode. The replay ignores
both incoming attacks and the initial pending meter, runs the full requested
horizon even if the recorded game ended earlier, and uses absorbing death bins
after a generated death. Existing garbage cells on the starting board remain.
The default horizon is eight; unknown future pieces are sampled from the bag.

This mode defaults to 4,000 thinks and one replay worker. It uses the original
entropy generator's search stopping configuration and queue RNG convention.
The legacy fixed-pressure mode retains its 1,000-think default. `--replay-weight`
must be positive to enable generation, but its numerical value does not weight
the synthetic examples: each generated continuation has equal supervised weight.
All continuations from the same source game stay together in train/validation/test.

No-garbage training keeps the cached continuations at each root together. Its loss
is sequence NLL (natural logs) plus `--moment-weight` (default 1) times the sum of:

- Squared error between predicted and target attack per piece, averaged over the horizon.
- Squared bucket-frequency error, summed over nine bins and averaged over offsets.

Targets are the per-root empirical frequencies of the cached continuations. Predictions
use eight sampled model prefixes per root, averaging the next-bin distributions rather
than only hard sampled bins. This is open-loop supervision: preceding attacks come from
the model itself. Score-function terms propagate gradients through prefix sampling;
they do not change the forward probability estimates. These finite-sample moment losses
have Monte Carlo noise. APP caps 7+ at 7 and assigns zero attack after death; bucket
frequencies retain the separate death bin. Validation uses fixed sampling draws and
selects the checkpoint using the same combined objective. Set `--moment-weight 0` for
the sequence-NLL control. Recorded-pressure training retains its existing objective.

Prepare targets **without training**, using a fresh output directory and a hard
memory limit on the remote machine (the example assumes the already measured
host with 15 GiB RAM; replay concurrency should be sized from measured peak RSS):

```sh
systemd-run --user --unit=nana-no-garbage-data -p MemoryMax=10G -p MemorySwapMax=1G \
  /path/to/python scripts/train_opponent.py \
  /root/stadium-collection/results/stadium-20261025-103813/stadium_*/games.db \
  --player auto --pieces --no-garbage --prepare-only --jobs 1 \
  --samples 64 \
  --exclude-positions results/cc-entropy-20261025/full-p2.jsonl \
  --cache results/no-garbage-data --out results/no-garbage-prepared
```

Use absolute script and data paths when invoking systemd from another directory.
`--exclude-positions` accepts a saved entropy experiment JSONL and removes its
root positions before sampling. It is repeatable and affects the cache key.
It excludes roots, not entire source games; use a separate source collection
for the external entropy benchmark, as in the example. Source database hashes,
replay executable, garbage mode, exclusions, generation settings, and loader
source all participate in cache identity. Completed shards survive interruption.
Preparation-only writes `data-manifest.json` and does not materialize all repeated
training examples or fit either model. No new training run has been launched.

When training is eventually requested, removing `--prepare-only` reuses those
shards. Each continuation currently becomes a separate example, repeating root
inputs; 64 continuations per root can substantially increase training RAM/time.
Model changes should be reviewed before that run. Architecture and old
checkpoints are unchanged by this data-pipeline option.

For a trained checkpoint, `measure_opponent_entropy.py` compares open-loop
per-offset forecasts and trajectory coverage against saved entropy continuations:

```sh
python scripts/measure_opponent_entropy.py --checkpoint PATH/cnn.pt \
  --rollouts results/cc-entropy-20261025/full-p2.jsonl --out comparison.json
```

This uses sampled model prefixes for future marginal probabilities and scores
complete target sequences exactly. It reports target entropy, model cross entropy,
attack probability errors, top-three trajectory/timing coverage, and held-queue
empirical candidate coverage. It does not train a model.

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
python scripts/train_opponent.py @gameDbs --player auto --replay-weight 0 --pieces --out results/cc-trajectories-real
```

Remove `--replay-weight 0` and choose another output directory to include CC replay supervision.

Default encoder: occupancy and two coordinate channels feed 32 independent heads (`nn.ModuleList`).
Each head has a four-channel stem, two residual blocks (two 3x3 convolutions each), a one-channel
1x1 projection, and flattening. No pooling or channel mixing between heads. Concatenate their
200-element vectors with context, then use one 256-unit linear/ReLU fusion layer. A 128-unit GRU
receives a 16-unit previous-emission embedding and the fusion vector at every step; a learned
projection of the fusion vector initializes its hidden state. A start token begins decoding.
`--heads`, `--width` (per head), `--blocks`, `--fusion`, `--hidden`, and `--embedding` configure sizes.
Heads are equivalent to independent groups in a grouped CNN, with a shared input stem per group.
They are separate modules to allow direct individual export and future worker scheduling.
Context is log1p(combo, B2B, meter). `--pieces` additionally supplies one-hot hold, current piece,
and five preview pieces; enable only if those are available to the intended predictor.
The context-only baseline uses the same fusion/GRU without board heads. The pooled baseline is a smoothed,
per-horizon categorical frequency distribution fitted only on training games.

Training uses `--samples N` (default 8) independent trajectories per root: each
draw samples a fresh legal hidden queue and performs one search run. Search
randomness varies naturally; it is not seeded by the CLI seed. Samples are cached
once and reused across epochs. There is no queue/repeat grouping in the training
targets. `cc_predictability.py` retains its separate Q x R sampling for uncertainty
diagnostics; that grouping is not a model uncertainty decomposition.

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
The loss is teacher-forced, recorded masked sequence NLL per observed piece plus `--replay-weight`
(default 0.5) times mean replay NLL. Each replay is decoded using its own preceding attacks;
the board encoding is shared across replays. Repeats at a position are averaged, rather than
increasing its weight. Death is absorbing during training and inference.
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
python scripts/train_opponent.py games.db --max-positions 128 --samples 2 --thinks 32 --epochs 2
```

For a useful first experiment, increase distinct positions and search budget, and use e.g. eight
independent trajectory samples. Datagen and training print progress separately. Tune
`--jobs` for replay and `--threads`/`--batch-size` for CPU training. Reducing `--width` or `--blocks`
offers cheaper encoder comparisons. Defaults are starting settings, not calibrated runtime promises.

Artifacts in `--out`:

- `report.json`: settings, source/data fingerprints, exact game splits, epoch timing/loss,
  untouched-test likelihood metrics at 1, 5, and 8 pieces (or the requested horizon),
  top-one/top-three exact sequence coverage, and mean retained probability mass.
- `cnn.pt`, `context.pt`: best validation checkpoints, including model settings.
- `heads/head-00.pt` through `head-31.pt`: individual head state dictionaries (default head count).
  The full checkpoint also contains fusion, initial-state, embedding, GRU, and output parameters.
- `test-predictions.npz`: teacher-forced conditional distributions, three decoded trajectories,
  their absolute log probabilities, real targets/masks, and game identifiers for analysis.

Offline candidate evaluation uses batched beam search (`--beam 32`) and retains three distinct
sequences. A finite beam approximates the top three; it does not guarantee finding them.
Exact coverage includes only fully observed windows. Sequence probabilities are products of the
autoregressive conditionals, not frequencies among samples, and are not renormalized over the
three retained sequences. Unretained mass is reported rather than assigned to them.
This trainer does not implement the future C++ parallel sampling/ranking runtime.

Positive `gain_vs_context` measures added information from board cells. Positive `gain_vs_pooled`
measures improvement over unconditional attack frequencies. Bits per piece use true preceding
attacks (teacher forcing), so summing the conditional log losses gives joint sequence NLL;
later-offset distributions and their Brier/bias diagnostics are conditional, not open-loop forecasts.
These gains are bits per piece; game-level
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
