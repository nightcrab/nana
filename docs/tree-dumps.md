# Policy training tree dumps

`nana_emulate --tree-dump-dir DIR` retains every search node during a game,
disables between-move tree collection, and streams `DIR/tree-SEED.jsonl` after
the last search stops. It then frees the tree before starting the next game.
Existing dump files are never overwritten; directory and write failures exit
with a nonzero status. Use separate seed ranges for independent workers.

For example, run eight processes with `--threads 1` and distinct `--seed` ranges:

```text
nana_emulate --threads 1 --games 100 --seed 0 --sims 1000 --search-seed 1 --max-pieces 50 --tree-dump-dir data/worker0
```

Each file starts with a versioned header containing the game seed, outcome,
rules, search options and evaluator configuration. The header's `min_node_N` is
10: nodes with N below 10 are skipped at export time. Every search node remains
in memory until the game ends. Each subsequent line is one qualifying node,
with all its candidate actions, including unvisited actions:

- `id`: the search's 32-bit node hash; unique within this game's retained tree.
- `N`: node visitation weight, exactly as stored by search.
- `board`: ten unsigned 32-bit columns; bit zero is the bottom row.
- `current`, `hold`, `queue`, `bag_left`, `b2b`, `combo`, `garbage_meter`,
  `phantom_meter`: the representative node state. Hold is null when empty.
- `belief`, when tracked: opponent belief marginals and freshness.
- `actions`: every candidate's type, rotation, x/y, spin, effective `N`, reward
  sum `R`, prior, static evaluation and move reward. Coordinates are absolute
  for the saved board; the search's garbage-height offset has been restored.

Piece IDs are S/Z/J/L/T/O/I = 0/1/2/3/4/5/6; Empty = 7. Rotations are
north/east/south/west = 0/1/2/3. Spins are none/mini/full = 0/1/2.

Construct visit targets by normalizing **edge** N within each node; node N need
not equal their sum. Counts retain the existing decay/timing semantics. Q for
visited actions is R/N. Unvisited actions have unknown Q.

The snapshot is the first state used to construct the retained node. Search
hashing abstracts garbage height and omits some context, including the full
preview; visits can therefore combine multiple contexts. A dump preserves that
existing search abstraction rather than recording every distinct full state.
It is a policy dataset, not a resumable search checkpoint.

State snapshots add memory and JSONL adds disk volume. Both are opt-in;
ordinary emulator runs retain their existing collection behavior.

Integration check:

```text
python scripts/test_tree_dump.py path/to/nana_emulate
```

## Fitting in C++

Build `nana_policy_fit`, then run:

```text
nana_policy_fit data fitted-policy 3
```

The optional last argument is the epoch count (default 3). The trainer streams
JSONL, reconstructs every placement with the engine, and checks that its feature
dot product reproduces the saved static evaluation within 0.00002. It fits 43
existing board and move features to normalized edge visits using softmax cross
entropy, weighted by node N. Jeopardy, T-slot and belief coefficients must be
zero in the input; they are excluded from this first fit. In particular, the
truncated jeopardy term is not linear in its coefficient.

Games with seed modulo 10 equal to zero form the validation set. Training files
are shuffled each epoch, with another shuffle over buffers of 2,048 nodes. Adam
uses batches of 128 nodes and L2 regularization of 0.00001. No feature dataset is
written. `policy.json` saves the best validation checkpoint, including feature
scales, logit coefficients and equivalent evaluator coefficients (`10000 * raw
logit coefficient`). `metrics.json` compares the fit with the stored original
prior, including N-weighted cross entropy, KL and top-action agreement.

These are **policy-only** coefficients. They have not been installed in search
or established as suitable leaf-value/reward weights by playing-strength tests.
Use an OS memory limit when running the trainer; the buffer keeps its working
set small but is not itself a process-wide memory limit.

```text
python scripts/test_policy_fit.py path/to/nana_emulate path/to/nana_policy_fit
```
