# Unified evaluation

Production uses `"eval": "nana"` (the default), or `"eval": "convex"` for the no-spin style.
Both styles share placement preparation and scoring. `EvalCC.cpp`, its API, and its
weights have been removed. `nana_eval_dump` dumps unified scores; `nana_cctree` retains
the experimental CC search but uses the unified evaluator. Neither tool provides
Cold Clear evaluation parity. Legacy `--terms`, `--positions`, and `--time` options
in `nana_eval_dump` are retired; scripts requiring those CC comparisons need the old version.

Nana retains its threshold penalties above heights 10 and 15, full-surface bumpiness,
leftmost-lowest well selection, unconditional well-column preference, and additive
clear/full-spin bonuses. Convex keeps its nonconvex rejection and prior handling.

Corrected board features:

- Row transitions count every horizontal boundary, including filled side walls, over
  all 32 rows. Empty rows contribute two transitions each.
- Cavities count all empty cells beneath column tops. Overhangs additionally count
  those cells with an open side, determined by either neighboring column's height.
  A cell accessible from both sides is counted once. Cavities and overhangs intentionally
  overlap; this preserves nana's all-hole penalty rather than CC's partition of holes.
- Covered cells count actual occupied cells above each buried hole. The squared feature
  sums the squared covering counts per hole. There is no six-cell cap.
- Counting detects four complete rows beside exactly one well above the lowest column's
  height. An empty board is not quad ready.
- Empty and full-height columns are handled without shifting 32-bit values by 32.
- Perfect clears contribute to move reward, so their bonus is not retained as board value
  during search backup.

New weights are `back_to_back`, `b2b_clear`, `combo_garbage`, `move_time`, `jeopardy`,
`well_depth`, `mini_tspins[0..2]`, and `tslot[0..3]`. They default to zero pending tuning.
`max_well_depth` defaults to 17; well depth counts consecutive complete rows beside the
selected well, below row 20. `combo_garbage` weights actual ruleset attack, including
combo/B2B/PC effects. Mini-spin bonuses are additive to ordinary clear rewards.

Move time follows CC's estimate: distance from column 4, minimum rotation count, one
for hold, and 40 for a clear. Jeopardy couples that time with height above ten.
Maintained B2B and opponent belief are position values; clear/B2B-clear/attack/time/
jeopardy rewards belong to the move. Belief uses the actual resulting attack and clear
status and requires `opponent: true` with nonzero `belief_weights`.

T-slot potential uses CC's pattern search and available-T estimate from the remaining
bag, preview, and hold. It simulates cutouts on a separate 40-row scratch board. Ordinary
board features always describe the real resulting board. Convex omits T-slot, wasted-T,
and spin bonuses, as appropriate for its no-spin style.

## Migration and checks

Old `eval: cc` and `cc_*` production configs must be migrated to nana's unprefixed keys.
Weights for similarly named features are not interchangeable because formulas differ.
The tuner space `tuner/spaces/eval.json` includes the new weights. Start a new tuning run:
old nana weights were tuned against the buggy features, and CC weights were tuned
against different definitions. Playing strength has not been established by regression tests.

Build with `-DNANA_EVALCHECK=ON`, then run the checks with the supplied nonzero-weight
fixture (PowerShell):

```powershell
$env:NANA_CONFIG = (Resolve-Path cpp/tests/evalcheck.json).Path
& out/build/<build>/cpp/nana_evalcheck.exe
```

The checks cover 2,000 boards against scalar feature calculations, reflection symmetry,
32-bit edge cases, clean-quad detection, reward decomposition, PC, B2B/combo, mini/full
spins, movement/jeopardy, T-slot potential, convex rejection, and opponent belief.
