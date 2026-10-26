# Exact-value tests through the real search

These tests exercise the production PUCT selection, worker/message queues,
virtual loss, MP-MCTS restarts, restart replay, flush messages, child tables,
and final move recommendation. An adapter replaces Tetris transitions and
leaf evaluation with finite stochastic trees whose values can be enumerated.
The normal bot does not compile the adapter or its state fields.

```sh
cmake -S . -B build-searchcheck -DCMAKE_BUILD_TYPE=Release -DNANA_GUI=OFF -DNANA_SEARCHCHECK=ON
cmake --build build-searchcheck --target nana_searchcheck -j4
# Use default config; custom eval weights/search settings change the experiment.
env -u NANA_CONFIG build-searchcheck/cpp/nana_searchcheck 12 > report.json
```

The executable runs 12 predeclared search seeds for each combination of six
trees, one/four workers, and 2,048/32,768 simulation budgets: 288 full searches.
Each report preserves per-seed root values, exact values, selected actions,
regret, root chance proposals, cached backup weights, and restart/replay counts.
There is no repeated sampling until a test passes.

## Questions the fixtures answer

| Fixture | Exact answer | What it checks |
|---|---|---|
| Forced fair coin | 0.5 | Full-path probability aggregation without deeper search |
| Forced rare, deep outcome | 0.125 | A 12.5% outcome must retain that probability despite five extra decision steps |
| Safe versus rare/deep | Safe action: 0.375 versus 0.125 | Wrong weighting must not turn an unprofitable gamble into a preferred move |
| Equal value, different fragmentation | Both actions: 0.6 | Deterministic versus 17 stochastic child subtrees |
| Seventeen improving children | 0.5 | Values must propagate from deeper search across the eight-child cache boundary |
| Profitable stochastic action | Gamble: 0.75 versus safe 0.6 | The search must also choose a gamble when it is actually better |

The lower budget is a screen. At 32,768 simulations the predeclared accuracy
requirement is absolute mean root-value error <=0.04 per action and wrong-move
fraction <=0.1 on unequal-value fixtures. Reports include across-seed standard
errors; these thresholds are numerical accuracy requirements, not p-values.

At BOTH budgets a structural invariant is mandatory: every cached root child
must be a possible immediate successor of that exact action. The fixture hashes
are unique node identifiers in an acyclic graph, so this is exact and has no
sampling uncertainty. Root proposals include samples made during restart replay;
proposal counts and backup counts are recorded separately, not equated blindly.

For sensitivity, run `nana_searchcheck 12 --inject-weights`. This deliberately
multiplies one coin outcome's backup weight by four without changing its draw
probability. It MUST exit 1; a clean exit means the harness failed to detect
the injected defect. A process/build failure (exit 2) is not a successful test.

## Bug reproduced and fixed, 2026-10-24

The first value-only run looked correct. Auditing the child identifiers revealed
that a restart replay which diverged carried the abandoned path's `child_hash`
into the new SELECT job. Ordinary selection overwrote that hash when descending
another edge. A terminal leaf or newly expanded leaf returned immediately,
leaving the stale hash attached to its reward. Its parent then recorded a
chance child which was never a successor of that action.

In the forced rare/deep fixture, about 11% of root backup mass was assigned to
impossible children in one search. Simple scalar means could still look right
because those incorrectly tagged outcomes shared the same payoff. Identity
checks failed in 17 of 24 configurations (288 individual searches).

The fix sets `job.child_hash = state.hash()` on entry to SELECT, before either
terminal or expansion handling. The returned value then identifies the state
actually reached, including after a diverged replay.

After the fix all 288 searches passed, with zero unexpected cached root children.
The injected-weight control still failed as required. In the one-worker rare/deep
case at 32,768 simulations, pooled proposal frequency was 0.124480 and backed-up
frequency 0.124345, against exact 0.125. The worst root-value RMSE among the
high-budget configurations was 0.0145 (rounded upward).

Raw reports and the failing baseline are under
`ec2-results/searchcheck-20261024/`: `before-fix.json`, `baseline.json`, and
`injected.json`. The pre-fix source archive is retained separately. The tests
establish the routing bug and its regression fix; they do not establish the
playing-strength impact of that fix.

## Limits

The adapter gives each node a unique hash and uses exact terminal payoffs. It
does not validate the HMM belief representation, lossy Tetris transpositions,
leaf evaluator, or depth-baseline correction. Tests of those require fixed real
positions with independently enumerated short-horizon outcomes. The earlier
[primitive statistical suite](statistical-tests.md) covers HMM draws separately.
