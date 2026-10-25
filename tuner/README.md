# nana's autotuner

Tunes nana's constants (eval weights, PUCT, the rollout score) by playing seeded games and running
particle swarm optimisation on the result. You give it a time budget; it measures how long games take
on this machine and tunes at the largest search budget that fits.

- **one budget knob**: `--budget-hours` picks the sims count, and the plan is printed before any
  tuning starts;
- **a fixed number of iterations** (15 by default), whatever the swarm does;
- **explore, then exploit**: the swarm starts widely scattered around the defaults and its inertia
  falls over the run, so it settles at the end;
- **a release gate**: `release.json` only changes when the tuned best beats the defaults in a
  sequential test on paired games;
- **checkpoints** every iteration, and **health checks** that catch a run that isn't measuring anything.

## Quick start

```
# nana_emulate built in Release; numpy installed
python -m tuner start eval1 --nana out/build/<build>/cpp/nana_emulate.exe --cpus 0,2,4,6,8,10,12,14 --budget-hours 8
python -m tuner status eval1          # from another terminal, any time
# Ctrl-C (or a reboot) stops it; then
python -m tuner resume eval1
```

Use one logical cpu per physical core (`--cpus`): each game runs single-threaded, pinned to its cpu,
and two games sharing a core would slow each other unevenly. Keep the machine otherwise idle.

To see what a budget buys without tuning, add `--plan-only`: it calibrates, prints the plan and
stops; `resume` then tunes. To check a setup end to end in a minute or two:

```
python -m tuner start smoke --nana ... --cpus ... --sims 64 --iterations 3 --particles 4 --games 4 --gate-games 16
```

## What a run does

1. **Health checks** (also on every resume), at 32 sims: nana loads exactly the configs written
   (`--print-config`), the same seeds replay the same games, and a deliberately broken eval (height
   penalties flipped) scores clearly worse.
2. **Calibration**: two waves of games over the cpus at each sims count of a ladder (256, 512, ...,
   131072), with the swarm's starting configs. From the seconds per game it projects the whole run,
   and picks the largest sims count that fits what's left of the budget. It stops climbing as soon as
   a rung won't fit (or clearly won't, at twice the last rung's cost). `--sims N` skips the choice.
   The measurements and the plan go to `plan.json`.
3. **Tuning**: `--iterations` PSO iterations at that sims count. A holdout (the best against the
   defaults on fixed seeds) every `--holdout-every` iterations and after the last.
4. **The gate**: the best against the release (the defaults) on fresh paired seeds, until an SPRT
   decides or `--gate-games` pairs are played. Only an accepted candidate becomes `release.json`.

The plan counts the gate at its limit, so most runs finish early. Games get longer as the swarm
improves (it survives longer), so iterations late in a run can take longer than calibration said;
`status` re-projects from the latest iterations.

## What a game is, and the fitness

`nana_emulate --threads 1 --sims N --search-seed 1` plays a game that replays exactly for a given
seed and config, so every config in an iteration plays the same seeds (common random numbers) and
luck in the pieces mostly cancels in comparisons. Games use TETR.IO rules (`--rules botris` plays
Botris; the ruleset is kept in `run.json`, `plan.json`, every `evals.jsonl` and `gates.jsonl` record
and `status.txt`, and the tuner stops if a game reports other rules than it asked for), the APP-only reward (`NANA_REWARD=app`), and the opponent HMM fitted on nana's stadium games as the
opponent, with the search modelling the same HMM.

Fitness is attack per piece over a fixed horizon (100 pieces): pieces lost by dying early score no
attack, so survival is part of the score.

The opponent's garbage is eased at small budgets, so a weak search survives long enough for the
fitness to say something: x0.25 at 32 sims, x0.6 at 512, x0.8 at 2048, x1.0 from 8192 up
(interpolated in log2 of the sims).

Configs reach nana through `NANA_CONFIG` (`cpp/Tetris/bot/Config.hpp`): a JSON file overriding any
default. nana rejects unknown keys, and `nana_emulate --print-config` shows what it loaded.

## Options

| option | default | |
|---|---|---|
| `--budget-hours H` / `--sims N` | (one is required) | the time budget, or a fixed sims count |
| `--iterations` | 15 | PSO iterations |
| `--particles` | 24 | swarm size; an iteration costs about 2 x particles x games games |
| `--games` | 32 | games per evaluation: more is less noise and slower iterations |
| `--spread` | 0.3 | the starting swarm's spread around the defaults (unit-cube units per parameter) |
| `--gate-games` | 400 | the gate's limit on paired games |
| `--holdout-every` | 5 | iterations between holdouts |
| `--rules` | `tetrio` | the ruleset games are played with: `tetrio` or `botris` |
| `--space` | `spaces/eval.json` | parameter space files (repeat to combine) |
| `-v`, `--verbose` | off | print everything; otherwise the console shows the plan, an ETA per iteration, each kind of warning once, and the result (`log.txt` always has everything) |
| `--overwrite` | off | replace a run of the same name without asking (`start` asks when it can) |

Ctrl-C stops a run at once: the running games are killed and the last finished iteration is kept, so
`resume` loses at most one iteration.

The inertia falls linearly from 0.9 to 0.4 over the iterations; the other PSO settings are 1.5, 1.5
and a speed limit of 0.2 (in `DEFAULTS`, `tuner/run.py`).

## Health warnings

Every iteration, in `status.txt` and the log:

- **signal**: the swarm's scores barely differ beyond one evaluation's noise: raise `--games`;
- **winner's curse**: the best's fresh scores keep falling below its average;
- **collapse**: the swarm has converged;
- **bounds**: most bests sit on a parameter's limit: widen it, or look for an exploit;
- **behaviour**: the best dies much more often than the defaults (APP bought with risk);
- **holdout**: beating the defaults on tuning seeds but not on held-out ones means overfitting.

## Parameter spaces (`tuner/spaces/`)

- `eval.json`: what the bot values. The rollout score's `app_weight` and `eval_weight`, and the
  33 eval weights nana reads (`clears[0]` and `tspins[0]` stay 0: the other entries are relative to
  them). The bounds are fixed, and are not moved when a tuning result becomes the default. Each
  term's sign is fixed by what it means: penalties in [-M, 0], bonuses (counting, perfect clear,
  tetris, t-spins) in [0, M], and well columns and 1-3 line clears in [-M, M]. M was set once as
  2.5 x the larger of the tuned value and Cold Clear's weight for the term (at least 20), rounded up;
  all ten well columns share +-150. A result near a bound means the bound was wrong: widen it there;
- `search.json`: how it searches. `c_init`, `c_base`, `n_cap`, `depth_step`, `prior_exponent`.

A space is a JSON list of `{"name", "default", "low", "high"}` with optional `"log": true` (uniform
in log space) and `"integer": true`; `name[i]` is one element of an array key.

`eval_weight` and the eval weights' overall scale are partly redundant: the eval is
1 + score / 10000, so both scale the score's share of the rollout, but only `eval_weight` scales
the constant 1 (what surviving is worth against dying, 0).

## A run's files (`tuner/runs/NAME/`)

| file | |
|---|---|
| `run.json` | settings: nana_emulate, cpus, space, options |
| `state.json` | checkpoint (optimiser, seeds, plan, release, baseline), written atomically every iteration |
| `plan.json` | calibration: seconds per game at each sims count tried, and the plan |
| `evals.jsonl` | every iteration: scores, the best and its values, warnings |
| `gates.jsonl` | the gate and its evidence |
| `release.json` | the released config (NANA_CONFIG file) |
| `best.json` | the swarm's current best (not gated) |
| `status.txt` | where the run is, rewritten every iteration |
| `log.txt` | everything printed |
| `configs/` | the config files the games used, by content hash |

## How the PSO copes with noise

A config's score is a mean over seeded games, so two evaluations of one config differ. Each
iteration plays one fresh block of seeds and evaluates on it both every particle's position and its
personal best: a position replaces its personal best only if it beats it on the same seeds, and a
personal best's score is the running mean of all its evaluations. The global best is the personal
best with the highest running mean among those evaluated at least twice. That costs about twice the
games of plain PSO and stops the swarm chasing lucky evaluations.

## Caveats

- The tuning budget is usually far below real play (about 100k sims). Eval weights should mostly
  transfer; search constants (`search.json`) may not.
- The fitness is APP, not wins. Stadium win rate is the final test of a release.
- The opponent HMM models nana at 1 pps; regenerate it (scripts/train_markov_opponent.py) when nana
  changes a lot.
- The ramp-based tuner this replaced (stages from 32 to 8192 sims) is in git history.
