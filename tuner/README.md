# nana's autotuner

Tunes nana's constants (eval weights, PUCT, the rollout score) by playing seeded games and running
particle swarm optimisation on the result. It's built to run unattended for days and be useful from
the first day:

- **a ramp**: cheap stages first (few simulations per move, eased opponent garbage), each warm-started
  from the last stage's best, up to budgets near real play;
- **releases**: `release.json` only changes when a candidate beats it in a sequential test on paired
  games, so whatever it holds is at least as good as the defaults, at any time;
- **checkpoints**: every iteration; resume after a crash or reboot, or restart the ramp at any stage;
- **health checks**: catch a run that isn't measuring anything before it wastes a week.

## Quick start

```
# nana_play built in Release (see notes/ for the build); numpy installed
python -m tuner start eval1 --nana out/build/markov/cpp/nana_play.exe --cpus 0,2,4,6,8,10,12,14
python -m tuner status eval1          # from another terminal, any time
# Ctrl-C (or a reboot) stops it; then
python -m tuner resume eval1
```

Use one logical cpu per physical core (`--cpus`): each game runs single-threaded, pinned to its cpu,
and two games sharing a core would slow each other unevenly. Keep the machine otherwise idle.

Check a setup first with the smoke schedule, a few minutes end to end:

```
python -m tuner start smoke --nana ... --cpus ... --schedule tuner/schedules/smoke.json --particles 8
```

## What a game is, and the fitness

`nana_play --threads 1 --sims N --search-seed 1` plays a game that replays exactly for a given
seed and config, so every config in an iteration plays the same seeds (common random numbers) and
luck in the pieces mostly cancels in comparisons. Games use TETR.IO rules (`NANA_RULES=tetrio`),
the APP-only reward (`NANA_REWARD=app`), and by default the opponent HMM fitted on nana's stadium
games as the opponent (`env: hmm`), with the search modelling the same HMM.

Fitness is attack per piece over a fixed horizon (100 pieces): pieces lost by dying early score no
attack, so survival is part of the score.

Configs reach nana through `NANA_CONFIG` (`cpp/Tetris/bot/Config.hpp`): a JSON file overriding any
default. nana rejects unknown keys, and `nana_play --print-config` shows what it loaded.

## The ramp (`tuner/schedules/week.json`)

| stage | sims/move | garbage | hours |
|---|---|---|---|
| s32 | 32 | x0.25 | 3 |
| s128 | 128 | x0.4 | 5 |
| s512 | 512 | x0.6 | 11 |
| s2048 | 2048 | x0.8 | 45 |
| s8192 | 8192 | x1.0 | 60 |

To fit the ramp to another budget, scale its timeline: every stage's hours and the gate interval
stretch or squeeze together, so the ramp keeps its shape.

```
python -m tuner start eval1 ... --total-hours 48      # the week in two days
python -m tuner start eval1 ... --time-scale 2        # twice as long everywhere
python -m tuner resume eval1 --total-hours 240        # change a running run's remaining stages
```

Squeezing doesn't make iterations cheaper, so a short timeline gives the high-budget stages few
iterations (at 48 h, s8192 gets about 23 h). The first gate moves with the scale: at `--total-hours
48` it comes after about 7 h.

`garbage_scale` eases the opponent's garbage: at full pressure a 32-simulation search dies in ~20
pieces whatever its eval, and the fitness says nothing. The first release gate comes at the end of
s512, about 20 hours in. Hours are tuning time; gates and checks add some.

Each stage keeps the swarm from the previous one, re-scattered around its best (`spread`). If a
stage over-specialised (its best doesn't hold up at the next budget), restart from an earlier best.

## Releases

At the end of every stage, and every `gate_every_hours`, the swarm's best plays the current release
on fresh paired seeds at the gate's budget (`gate` in the schedule; once the ramp passes it, the
stage's own) until an SPRT decides: H1 = better by `delta` (2% of the release's fitness), at
alpha = beta = 0.05, at most `max_games` pairs. Only an accepted candidate becomes the release.
`gates.jsonl` keeps every gate.

`release.json` is a complete NANA_CONFIG file: `NANA_CONFIG=tuner/runs/NAME/release.json` runs nana
with it (tbp_tetris included).

## Health checks

Before tuning (and on every resume), stopping the run on failure:

- **plumbing**: nana loads exactly the values written (`--print-config`);
- **determinism**: the same seeds replay the same games;
- **canary**: a deliberately broken eval (height penalties flipped) scores clearly worse.

Every iteration, warnings in `status.txt` and the log:

- **signal**: the swarm's scores barely differ beyond one evaluation's noise: raise `games_per_eval`;
- **winner's curse**: the best's fresh scores keep falling below its average;
- **collapse**: the swarm has converged;
- **bounds**: most bests sit on a parameter's limit: widen it, or look for an exploit;
- **behaviour**: the best dies much more often than the defaults (APP bought with risk);
- **holdout**: every `holdout.every` iterations the best plays fixed held-out seeds against the
  defaults; beating the defaults on tuning seeds but not on held-out ones means overfitting.

## Restarts and changes

```
python -m tuner restart-stage NAME K [--from release|best|CONFIG.json]
python -m tuner add-params NAME tuner/spaces/search.json
python -m tuner check NAME              # health checks only
```

`restart-stage` reruns the ramp from stage K (0-based), the swarm scattered around the chosen
config; the release is kept. `add-params` adds parameters (new ones start at their defaults), e.g.
the PUCT and rollout weights once the eval has settled.

## Parameter spaces (`tuner/spaces/`)

- `eval.json`: the 33 eval weights nana reads, each bounded to default +- max(|default|, 20)
  (`clears[0]` and `tspins[0]` stay 0: the other entries are relative to them);
- `search.json`: `c_init`, `c_base`, `depth_step`, `prior_exponent`, `app_weight`, `eval_weight`.

A space is a JSON list of `{"name", "default", "low", "high"}` with optional `"log": true` (uniform
in log space) and `"integer": true`; `name[i]` is one element of an array key.

Tuning `eval_weight` together with the eval weights is partly redundant (the eval's overall scale
and its weight in the rollout score trade off), so `search.json` is meant for later.

## A run's files (`tuner/runs/NAME/`)

| file | |
|---|---|
| `run.json` | settings: nana_play, cpus, space, schedule |
| `state.json` | checkpoint (optimiser, seeds, release, baselines), written atomically every iteration |
| `evals.jsonl` | every iteration: scores, the best and its values, warnings |
| `gates.jsonl` | every gate and its evidence |
| `release.json` | the released config (NANA_CONFIG file) |
| `best.json` | the swarm's current best (not gated) |
| `status.txt` | where the run is, rewritten every iteration |
| `log.txt` | everything printed |
| `configs/` | the config files the games used, by content hash |

## How the PSO copes with noise

A config's score is a mean over 16 seeded games, so two evaluations of one config differ. Each
iteration plays one fresh block of seeds and evaluates on it both every particle's position and its
personal best: a position replaces its personal best only if it beats it on the same seeds, and a
personal best's score is the running mean of all its evaluations. The global best is the personal
best with the highest running mean among those evaluated at least twice. That costs about twice the
games of plain PSO and stops the swarm chasing lucky evaluations.

## Caveats

- Gains at tiny budgets can be specialisation: the ramp and the gates at higher budgets are the guard.
- The fitness is APP, not wins. Stadium win rate is the final test of a release.
- The `hmm` opponent models nana at 1 pps; regenerate it (scripts/train_markov_opponent.py) when nana
  changes a lot.
