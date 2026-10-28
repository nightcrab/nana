# Budgeted policy iteration

`scripts/policy_iterate.py` repeatedly collects search nodes, fits a policy for
exactly two epochs, logs paired strength, and selects that fitted checkpoint as
the next iteration's datagen policy. The ordinary value/reward configuration is
fixed throughout. Production defaults are never edited.

```sh
python3 scripts/policy_iterate.py \
  --nana build/cpp/nana_emulate --trainer build/cpp/nana_policy_fit \
  --config tuner/releases/unified-cc-20261028.json \
  --initial-model tuner/releases/trained-policy-20261028.json \
  --out results/policy-iteration --budget-hours 4 \
  --cpus 0,2,4,8,10,12 --data-gb 90
```

Without `--initial-model`, the first collection uses the rank prior and the first
fit starts at zero. With it, fits warm-start from the incumbent scaled weights;
optimizer moments reset for each new dataset. Each fit completes epochs 1 and 2
and saves the best validation checkpoint, including the incumbent at epoch 0.

Each cycle defaults to:

1. Up to 1,000 independent games, 50 pieces, 1,000 sims/move, one thread per CPU.
   Only nodes with N >= 10 are exported. The incumbent policy generates visits.
2. A streamed C++ fit with no intermediate feature dataset. Games with seed
   modulo ten equal to zero are held out. The fit subprocess has a hard
   2,000,000,000-byte address-space limit on POSIX; Windows local smoke tests can
   explicitly disable this with `--fit-memory-bytes 0` under another memory limit.
   Jeopardy and T-slot terms may be enabled in the fixed value evaluator: the
   trainer loads the exact datagen config to check their saved evaluations.
   Belief weights must be zero because the dumps do not contain joint beliefs.
3. 100 fresh strength seeds, 100 pieces, comparing candidate, incumbent and the
   frozen rank prior with identical value settings. Default strength budget is
   the datagen simulation count; `--strength-ms 50` selects fixed-time testing.
   Arm order rotates. Logs include attack/fixed horizon, survival, throughput,
   paired differences and approximate 95% confidence intervals.
4. Commit the checkpoint and metrics, then clear and reuse the raw-data buffer.

Strength is logged, not used as a promotion gate. Only a cycle that completes
collection, both fit epochs and strength testing advances the experiment's
policy. An incomplete final cycle never replaces the last committed policy.

## Buffer and deadline

The single `OUT/data` directory is capped at **90,000,000,000 bytes** by default.
Workers reserve space before launching a game, and the C++ exporter enforces its
reservation using `--tree-dump-max-bytes`. File allocation is block-rounded when
accounting reservations. Capped game files are discarded rather than training
on a prefix selected by unordered-map order. Collection stops at its game count
or buffer limit. Models, metrics and small game logs are retained separately.

The buffer is **replaced**, not appended to or archived, between iterations. It
has an ownership marker; symlinks and unowned directories are refused before
recursive removal. A new run requires a new output directory. `--resume` checks
the original options and input hashes, recovers completed commits, discards
unfinished raw data when beginning another cycle, and continues within the
original deadline. An exclusive run lock prevents concurrent writers. Downtime
counts toward a resumed run's budget.

Every subprocess is checked against the common monotonic deadline; expiry
terminates in-flight games or fitting. `--max-iterations` is an optional extra
stop condition for smoke tests. The four-hour iteration budget starts when the
iterator starts, excluding any external queue wait or build.

## Outputs

- `status.json`: current phase, counts, deadline remaining and committed model.
- `history.jsonl`: one record per completed iteration, with data/fit/strength metrics.
- `iteration-NNNNN/`: teacher config, fit checkpoint, raw strength results and logs.
- `latest.json`: path and hash of the latest committed policy.
- `data/`: only the latest iteration's raw dumps, including unfinished data at expiry.

Regression checks:

```text
python scripts/test_policy_iterate.py
python scripts/test_tree_dump.py path/to/nana_emulate
```
