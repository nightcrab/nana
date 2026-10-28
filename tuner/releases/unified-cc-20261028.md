# Unified evaluator release, 2026-10-28

`unified-cc-20261028.json` is the released config downloaded from the completed
`unified-cc-8h` run in `/root/nana-unified-cc-tune-20261027`.

- Source commit: `ef9ef7e136c0f6bc1b3b3d70cc8dcd7be16f578f` (snapshot included local engine changes).
- Build: Release, `NANA_CC_SEARCH=1`, native CPU tuning, mimalloc disabled.
- Evaluator: unified nana; Tetrio rules; opponent modeling disabled.
- Resources: six physical cores, one thread per game; eight-hour budget.
- Optimizer: PSO, 24 particles, 15 iterations, 45 evaluation parameters.
- Conditions: 1,024 simulations, garbage scale 0.7, 100-piece horizon.
- Final holdout: 0.8450 attack/piece versus 0.2834375 for the old defaults.
- Release gate: accepted after 24 paired games, 0.8379166666666666 attack/piece
  versus 0.2875 for the old defaults (LLR 7.03).
- Finished: 2026-10-28 05:03:30 JST, after 6.25 hours.

The 45 tuned parameters are applied to `cpp/Tetris/bot/Config.hpp` and the matching
starting defaults in `tuner/spaces/eval.json`. Search and runtime defaults were not
part of this tune. These results measure the CC search under the conditions above;
other search modes, rules, and simulation budgets were not independently gated.
