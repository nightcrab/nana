# nana's autotuner

Tunes nana's constants (eval weights, PUCT, the rollout score) by playing seeded games and running
PSO (the default), CMA-ES, or SPSA on the result. You give it a time budget; it measures how long games take
on this machine and tunes at the largest search budget that fits.

- **one budget knob**: `--budget-hours` picks the sims count, and the plan is printed before any
  tuning starts;
- **a fixed number of iterations** (15 by default);
- **adaptive exploration**: PSO reduces inertia over the run; CMA-ES learns a covariance matrix
  and adapts its step size, starting from the defaults;
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

To use CMA-ES with the existing evaluation/game infrastructure:

```
python -m tuner start cma1 --nana out/build/<build>/cpp/nana_emulate.exe --cpus 0,2,4,6 --budget-hours 8 --optimizer cmaes --particles 24 --spread 0.3
```

CMA-ES uses NumPy only, with no additional dependency. `--particles` is its population size
(at least 2); `--spread` is its initial standard deviation in normalized parameter space
(finite and positive). Both default to the same values as PSO. Resume restores the optimizer
and its random state from the checkpoint.

To use SPSA (simultaneous perturbation stochastic approximation):

```
python -m tuner start spsa1 --nana out/build/<build>/cpp/nana_emulate.exe --cpus 0,2,4,6 --budget-hours 8 --optimizer spsa --iterations 200 --spsa-learning-rate 0.05 --spsa-perturbation 0.1
```

SPSA changes all parameters along a random sign direction and estimates a gradient from two
perturbed configs. Each iteration evaluates those two configs and the incumbent on the same
fresh seeds: **3 x games** evaluations, independent of the parameter count. It usually needs
more iterations than the population optimizers. `--particles` and `--spread` do not affect SPSA.
The incumbent is the best evaluated config, promoted using fresh paired scores; the internal
gradient update is evaluated through the next iteration's perturbations. Checkpoint/resume,
calibration, holdouts, and the release gate work as with the other optimizers.

For iteration `k` starting at 1, the learning rate is `a / (k + A)^alpha` and the perturbation
is `c / k^gamma`. The following options are saved in `run.json` and reused on resume:

| Option | Default | Meaning |
| --- | --- | --- |
| `--spsa-learning-rate` | 0.05 | `a`, gradient learning-rate coefficient |
| `--spsa-perturbation` | 0.1 | `c`, initial perturbation in normalized parameter units; at most 0.5 |
| `--spsa-stability` | 10 | `A`, nonnegative learning-rate offset |
| `--spsa-alpha` | 0.602 | learning-rate decay exponent, in (0, 1] |
| `--spsa-gamma` | 0.101 | perturbation decay exponent, in (0, 1] |
| `--spsa-max-step` | 0.1 | maximum update per parameter in normalized units; at most 1 |
| `--spsa-start-jitter` | 0.05 | starting offset per parameter in normalized units, random sign; 0 to disable; at most 0.5 |

SPSA starts from the defaults moved by the start jitter in every parameter. The two probes are
mirror images, so where the gradient at the defaults is zero (for example, products of weights that
all default to zero, as in `conditional.json`) both probes score the same and the center never
moves; the offset avoids that. The incumbent starts at the unmodified defaults.

All coefficients must be finite and positive except the stability offset and start jitter, which may be zero.
Probes and updated parameters are clipped to [0, 1]; the gradient uses the actual probe
separation at boundaries. Unlike PSO and CMA-ES, SPSA's learning rate depends on the objective's
score scale. Reduce it for large scores (especially sims/s or game-score) if updates repeatedly
hit the step cap. Increase `--games` when paired differences are noisy. For integer parameters,
choose perturbations large enough to produce distinct rounded configs.

Use one logical cpu per physical core (`--cpus`). By default each game uses one thread, pinned to
its cpu. `--threads N` assigns N search threads and N CPUs to each game, taking consecutive groups
from the supplied CPU list. The CPU count must be divisible by N, with no duplicate CPUs; for
example, `--cpus 0,2,4,6,8,10 --threads 3` runs two games concurrently on groups `0,2,4` and
`6,8,10`. Choose groups across L3 caches if you want to measure cross-cache stealing. Keep the
machine otherwise idle. The thread count is saved and reused on resume; older runs use one thread.

To optimise search throughput instead of strength:

```
python -m tuner start speed1 --nana out/build/<build>/cpp/nana_emulate.exe --cpus 0,2,4,6 --sims 8192 --objective sims-per-second
```

This defaults to `spaces/search.json`, 4 games per evaluation (instead of 32), 8 holdout games
(instead of 32), and at most 32 release-gate pairs (instead of 400). `--space`, `--games`, and
`--gate-games` still override their defaults. All optimizers and the holdouts/release gate maximise
sims/s: `1000 * sims_per_move / ms_per_move`, averaged across games. This measures search time,
including search start/stop overhead, without process startup. It does not reward dying early by
counting shorter games as faster. Timing varies, so increase `--games` if signal warnings persist;
use a substantial sims count to avoid measuring mostly overhead. Speed gains can sacrifice strength.
Single-threaded seed determinism is checked using the selected objective. The canary and death-rate warning are
skipped in speed mode. Existing runs resume with their original objective (strength for older runs).

To tune work stealing with three threads per game (rebuild `nana_emulate` for CPU-group support):

```
python -m tuner start stealing1 --nana out/build/<build>/cpp/nana_emulate.exe --cpus 0,2,4,6,8,10 --threads 3 --sims 8192 --objective sims-per-second --space tuner/spaces/work_stealing.json
```

Multithreaded search depends on scheduling, so exact seed replay is skipped for these runs.
Config checks, throughput validation in speed mode, and the strength canary in strength mode still run.

To see what a budget buys without tuning, add `--plan-only`: it calibrates, prints the plan and
stops; `resume` then tunes. To check a setup end to end in a minute or two:

```
python -m tuner start smoke --nana ... --cpus ... --sims 64 --iterations 3 --particles 4 --games 4 --gate-games 16
```

## What a run does

1. **Health checks** (also on every resume), at 32 sims: nana loads exactly the configs written
   (`--print-config`), the same seeds replay the same games with one thread, and a deliberately broken eval (height
   penalties flipped) scores clearly worse.
2. **Calibration**: two waves of games over the cpus at each sims count of a ladder (256, 512, ...,
   131072), with the optimizer's starting configs. From the seconds per game it projects the whole run,
   and picks the largest sims count that fits what's left of the budget. It stops climbing as soon as
   a rung won't fit (or clearly won't, at twice the last rung's cost). `--sims N` skips the choice.
   The measurements and the plan go to `plan.json`.
3. **Tuning**: `--iterations` optimizer iterations at that sims count. A holdout (the best against the
   defaults on fixed seeds) every `--holdout-every` iterations and after the last.
4. **The gate**: the best against the release (the defaults) on fresh paired seeds, until an SPRT
   decides or `--gate-games` pairs are played. Only an accepted candidate becomes `release.json`.

The plan counts the gate at its limit, so most runs finish early. Games get longer as the bot
improves (it survives longer), so iterations late in a run can take longer than calibration said;
`status` re-projects from the latest iterations.

## What a game is, and the fitness

### Stadium win rate against the baseline

Use `--objective stadium` for head-to-head games against the fixed starting configuration
stored in `state.json` as `base_config`:

```sh
python -m tuner start versus1 --nana build/cpp/nana_emulate --bot build/cpp/nana_tbp --stadium stadium/src/stadium_cli --cpus 0,2,4,6 --threads 1 --sims 8192 --objective stadium
```

Build [Universal-Shaktris-Stadium](https://github.com/nightcrab/Universal-Shaktris-Stadium)
with support for the `threads`, `thinks`, and `garbage_delay` settings, as used by
`scripts/opponent/stadium_collect.sh`. `--nana` still supplies the canonical config;
`--bot` must be a matching `nana_tbp` build. Each match gets isolated executable copies and
`nana.json` files, leaving the originals untouched. Windows DLLs beside the bot are copied too.

Fitness is 1 for a win, 0.5 for a draw, and 0 for a loss. The baseline score is fixed
at **0.5**; no noisy self-play sample is
used as the reference. Startup logs and status show the reference 95% Wilson confidence
interval at a 50% score for the configured evaluation and holdout game counts. For example,
64 games gives about **38.1%–61.9%**, and 32 games gives **33.6%–66.4%**. These describe
finite-sample uncertainty assuming independent win/loss outcomes; draws reduce variance.
They are reference intervals, not confidence bounds for a selected best candidate.
Resuming an older stadium checkpoint replaces its sampled baseline with 0.5.

Both bots receive the selected simulation budget, sides alternate, and each concurrent
match reserves `2 * --threads`
CPUs for the two bots; on Linux each bot is pinned to its own CPU group. The CPU count
must be divisible by that number. Matches run
to completion, without the emulator's horizon or eased garbage. Botris selects one-turn
garbage delay; TETR.IO selects zero delay.

The upstream stadium has no seed interface: seed counters select sides and track work,
but do not replay games or provide common random numbers. Holdouts are fresh independent
matches, and exact replay and the small-sample canary check are skipped. The final gate
keeps the existing approximate Gaussian SPRT, comparing candidate and release match scores
against the same fixed baseline, using independent games. Its defaults remain a 2% relative
improvement, alpha/beta 0.05, and at most 400 games per config. This is not a dedicated
binomial win/loss SPRT; reaching the limit without a decision remains inconclusive.

### Emulator objectives

`nana_emulate --threads 1 --sims N --search-seed 1` plays a game that replays exactly for a given
seed and config, so every config in an iteration plays the same seeds (common random numbers) and
luck in the pieces mostly cancels in comparisons. Games use TETR.IO rules (`--rules botris` plays
Botris; the ruleset is kept in `run.json`, `plan.json`, every `evals.jsonl` and `gates.jsonl` record
and `status.txt`, and the tuner stops if a game reports other rules than it asked for), and the opponent HMM fitted on nana's stadium games as the
opponent, with the search modelling the same HMM.

Fitness is attack per piece over a fixed horizon (100 pieces): pieces lost by dying early score no
attack, so survival is part of the score.

To optimise the emulator's `game_score` field instead, use 
`--objective game-score`. Its calculation in `EmulationGame` 
with the scoring rules you want before relying on it as a final
Tetris score:

```
python -m tuner start score1 --nana build/cpp/nana_emulate.exe --cpus 0,2,4,6 --sims 8192 --objective game-score
```

The opponent's garbage is eased at small budgets, so a weak search survives long enough for the
fitness to say something: x0.25 at 32 sims, x0.6 at 512, x0.8 at 2048, x1.0 from 8192 up
(interpolated in log2 of the sims).

Configs reach nana through `NANA_CONFIG` (`cpp/Tetris/bot/Config.hpp`): a JSON file overriding any
default. nana rejects unknown keys, and `nana_emulate --print-config` shows what it loaded.

## Options

| option | default | |
|---|---|---|
| `--budget-hours H` / `--sims N` | (one is required) | the time budget, or a fixed sims count |
| `--objective` | `strength` | maximise attack/piece, `game-score`, `stadium` win rate, or `sims-per-second` (with the smaller defaults above) |
| `--optimizer` | `pso` | optimizer: `pso`, `cmaes`, or `spsa` |
| `--threads` | 1 | search threads per game; must divide the CPU count (1–64) |
| `--iterations` | 15 | optimizer iterations |
| `--particles` | 24 | swarm/population size; PSO costs about 2 x particles x games per iteration, CMA-ES costs (particles + 1) x games; SPSA ignores this and costs 3 x games |
| `--games` | 32 | games per evaluation: more is less noise and slower iterations |
| `--spread` | 0.3 | initial spread around defaults in unit-cube units; CMA-ES initial sigma |
| `--gate-games` | 400 | the gate's limit on paired games |
| `--holdout-every` | 5 | iterations between holdouts |
| `--rules` | `tetrio` | the ruleset games are played with: `tetrio` or `botris` |
| `--eval` | executable default (`cc`) | evaluator: `cc`, `nana`, or `convex`; saved in the run's base config |
| `--space` | `spaces/eval.json` (speed: `spaces/search.json`) | parameter space files (repeat to combine) |
| `-v`, `--verbose` | off | print everything; otherwise the console shows the plan, an ETA per iteration, each kind of warning once, and the result (`log.txt` always has everything) |
| `--overwrite` | off | replace a run of the same name without asking (`start` asks when it can) |

Ctrl-C stops a run at once: the running games are killed and the last finished iteration is kept, so
`resume` loses at most one iteration.

The inertia falls linearly from 0.9 to 0.4 over the iterations; the other PSO settings are 1.5, 1.5
and a speed limit of 0.2 (in `PSO_DEFAULTS`, `tuner/pso.py`).

## Optimizer interface

Search priors rank moves by their cached leaf evaluation and assign normalized
`rank ** -prior_exponent` weights. Tune the leaf weights with `spaces/eval.json`
and the rank power with `spaces/search.json`. Use PUCT search (`-DNANA_CC_SEARCH=0`,
the default) for these priors.

`tuner/optimizer.py` defines the optimizer contract and creation/restore functions. `Run` owns
parameter decoding, game execution, shared seed blocks, generic health checks, holdouts, and the
release gate. It uses opaque candidate IDs and does not inspect optimizer internals.

An optimizer supplies:

- `ask()`: candidates in normalized [0, 1] space, including requested reevaluations. This call
  does not advance the optimizer; all candidates receive the same fresh seeds.
- `tell(scores)`: per-game scores keyed by candidate ID, with larger scores preferred. It advances
  the optimizer and returns candidate scores for monitoring, any fresh score of the previous best,
  and the candidate ID providing game evidence for the current recommendation.
- `best()`: a recommendation with its unit vector, mean, evaluation count, and an opaque index.
- `sample_points()` and `evaluation_counts(iterations)`: representative calibration configs and
  candidate counts for projecting the cost of a complete run.
- `diagnostics()` and `warnings(parameter_names)`: algorithm-specific metrics and warnings.
- `state()`: a JSON checkpoint with optimizer name, state format version, and algorithm state.

`PSOOptimizer` in `tuner/pso.py` adapts the original PSO implementation, including its inertia
schedule, paired personal-best reevaluations, and swarm diagnostics. To add an optimizer, implement
the contract and extend the defaults, creation, restore, and CLI choices in `tuner/optimizer.py`.

`CMAESOptimizer` in `tuner/cmaes.py` implements positive-weight full-covariance CMA-ES with
rank-one and rank-mu covariance updates and cumulative step-size adaptation, using the learning
rates from [Hansen's tutorial](https://arxiv.org/abs/1604.00772). Its Gaussian operates in unbounded
coordinates; a periodic reflection maps candidates into [0, 1] before the existing parameter
decoder applies bounds, log scales, and integer rounding. It supports one-dimensional spaces too.

Each CMA-ES generation evaluates the population plus the incumbent (initially the defaults) on
the same fresh seeds. The generation winner replaces the incumbent only if its fresh mean beats
the incumbent's fresh mean. A retained incumbent accumulates an average across evaluations;
a replacement starts a new average. Only the population updates the search distribution.
This provides paired comparisons for the final recommendation; covariance adaptation still uses
noisy population rankings, so increase `--games` when signal is weak. The fixed iteration limit,
holdouts, and release gate apply to all optimizers. CMA-ES reports sigma, covariance condition,
and population diversity, with warnings for tiny steps, poor conditioning, and a best near bounds.

Existing checkpoints with a bare `pso` field still resume. The next save writes the named,
versioned `optimizer` envelope; optimizer settings remain in `run.json`. Existing command lines
continue to select PSO by default, and `--particles` and `--spread` retain their meanings.

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
- `opponent_conditioning.json`: all 67 opponent-conditioning weights: `opp_cond_u[0..18]`
  (the opponent gate) and `opp_cond_v[0..47]` (the placement direction), in the enum order in
  `cpp/Tetris/bot/OpponentConditioning.hpp`. Defaults are zero, matching the disabled term in
  `NanaConfig`; `opp_cond_mean` stays at the base config's fixed values. Initial bounds are
  symmetric: gate bounds scale inversely with feature magnitude, and placement bounds use
  the corresponding eval-space magnitudes, allowing either sign of adjustment. These are
  starting ranges, not measured limits. Use `--space tuner/spaces/opponent_conditioning.json`
  with `--objective stadium`, `--bot`, and `--stadium`: the TBP bot sets the gate from the
  visible opponent, while the emulator does not supply this state. Use `--eval nana` to
  exercise spin features too (`convex` zeros them). Repeat `--space` to tune eval weights
  alongside conditioning.
- `work_stealing.json`: all eight work-stealing settings: `message_weight`, `rollout_weight`,
  `low_water`, `give_margin`, `backoff_iterations`, `steal_chiplet_penalty`, `steal_topk`, and
  `steal_temperature`. Integer bounds include the current config defaults; `steal_topk` spans
  its supported range of 1–8. Select it with `--space tuner/spaces/work_stealing.json`.
  Use `--threads` greater than 1 to measure work stealing.

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

