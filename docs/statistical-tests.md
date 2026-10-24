# Garbage sampling and backup diagnostics

Build the optional target in a normal sampled, suffix-expectimax configuration:

```sh
cmake -S . -B build-statcheck -DCMAKE_BUILD_TYPE=Release -DNANA_GUI=OFF -DNANA_STATCHECK=ON
cmake --build build-statcheck --target nana_statcheck -j4
python scripts/statcheck.py --binary build-statcheck/cpp/nana_statcheck --out results/statcheck
```

This calls production `MarkovOpponent`, `EmulationGame::start_from_root`,
`ChildTable`, `UCTNode::report`, `stepUp` and `rewardScalar`. It does not run
strength games or require a Python statistics package. Diagnostic builds with
the future oracle enabled are rejected at compile time.

## Pass/fail checks

* **One-turn emissions:** exact sums over all hidden transitions and emissions,
  compared with sampled frequencies in 13 contexts. Contexts include fresh and
  established opponents, inputs 0..5, jeopardy buckets and since buckets.
* **Two-turn joint emissions:** explicitly enumerate both hidden transitions,
  emission-dependent since counters, meter cancellation/tanking, and death/revival.
  Check all 81 pairs, retaining temporal correlations rather than multiplying
  marginal probabilities. Verify delivered damage separately on each trial.
* **Restart seeds:** fresh-root emissions and successive independently reseeded
  emissions. The latter checks for correlations introduced by the seeding path.
* **Root sampling:** call the real `start_from_root` and `chance_move` after
  1, 8, 24, and 64 unobserved turns, and check against the root representation's
  stated independent-marginal distribution.
* **Known chance values:** check empirical probability-weighted backups with
  2, 8, and 17 children, covering both cached and overflow children.
* **Tiny decision tree:** a risky action has exact value 0.25 and a safe action
  value 0.4. Verify the correct action, then update a child to change the risky
  action to 0.84 and verify propagation. Lower it again and verify best-arm rescan.

Use 100,000 trials per histogram by default and three seeds fixed before running.
For categorical frequencies, two-sided Bernstein concentration bounds use
`sqrt(2*p*(1-p)*log(2*M/alpha)/N) + 2*log(2*M/alpha)/(3*N)`.
Bounded scalar averages use Hoeffding bounds. `M=10,000` conservatively covers
the whole three-seed run, and family alpha is 0.001. Do not keep drawing samples
until a check passes. All outputs, including failures, are retained.

These bounds assume independent trials from the specified distribution. Actual
draws are deterministic pseudorandom streams; the tests are reproducible checks
for discrepancies, not a proof of independence or a universal RNG certification.
Passing a screen excludes only effects large enough for its sample size and
does not validate very rare tails or all possible HMM contexts.

## Diagnostics without a pass/fail verdict

These fixtures expose assumptions with exact counterexamples. They do not claim
the same conditions arise in a real search, nor that every difference is a bug.

* **Outcome-dependent batch weights:** a 10% chance outcome has value 1 and
  takes ten units of descendant work; others take one. Counting draws gives 0.1;
  weighting by work gives 10/19. Production `report` accepts a batch count `k`,
  so the scheduler must ensure it represents chance draws rather than effort.
* **Overflow refresh:** the first eight children replace their old value with
  the latest. Overflow children accumulate their historical values instead.
  Their values can therefore react differently when deeper search improves them.
* **Merged child values:** feeding varying state-specific samples under one hash
  retains the last value, not their mean. Correctness requires the reported value
  to estimate the shared child's conditional expectation.
* **Reward objective:** pooled attack/length differs from the expected per-path
  attack rate when path lengths differ. A two-path fixture gives 0.1 versus 0.5.
* **Root factorization:** one unobserved attack correlates hidden state and time
  since last attack. Compare the filter's exact joint prediction with the product
  of marginals actually sampled by the root. Report state and emission total variation.

## Remaining integration work

The [full-search integration fixtures](search-integration-tests.md) now cover
the scheduler on exact-value trees, including proposal/backup audits. They found
and reproduced a stale child-identity bug after diverged restart replay. The
remaining real-position audit described below is still useful beyond those trees.

This suite tests the sampler and backup primitives, not the full asynchronous
scheduler, transposition graph or depth baseline. The next integration audit
should record fresh chance proposals, accepted/rejected restart replays, and
weighted completions separately for each parent action/child. Compare counts
after draining pending work. Equal total counts alone do not establish correct
conditional probabilities. Include a forced-action fixture with independently
enumerated chance probabilities, so adaptive action selection cannot hide a
weighting discrepancy. No production search behavior is changed by this suite.

## First EC2 run, 2026-10-24

All 153 pass/fail checks passed: 51 per seed, 100,000 trials per histogram.
Reports are saved under `ec2-results/statcheck-20261024/`. No discrepancy was
detected in the tested HMM draws, two-turn correlations, cancellation, restart
seeding, or fixed-value chance backups at this sensitivity.

The root approximation changes the next-emission distribution by 0.0656 total
variation after eight unobserved turns in the fixture (0.0208 after 24 turns,
0.0205 after 64). This is a deterministic difference between distributions,
not sampling error or an estimate of playing-strength impact.

The exact diagnostic fixtures reproduce overflow staleness (0.8556 backed up
versus 0.9 if every child used its latest value), last-value replacement under
a merged hash (1.0 versus sample mean 0.5), and effort-weighted outcomes (0.5263
versus probability-weighted 0.1). These identify conditions worth auditing in
the full search; their magnitudes are specific to the constructed fixtures.
