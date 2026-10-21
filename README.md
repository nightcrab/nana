<p align="center">
<img class="only-light" height="150px" src="./logo.png"/>
</p>
<br>

# Nana

--------------------------

Nana is a parallel search program for two-player stacking games, such as Versus Tetris. 

Versus Tetris presents a variety of challenges that Nana chooses to approach explicitly, namely:

* Large branching factors
* Random chance
* Hidden information
* Efficient parallelisation
* Online any-time planning

The engine is designed to run on tournament hardware with an emphasis on high core counts and modern x86.


## How It Works

Nana at its core is based on the combination of two search techniques; Monte Carlo Tree Search (MCTS)[[1]](#1), and Transposition Driven Scheduling (TDS)[[2]](#2).

MCTS operates on the premise that we can generate an analysis of the game's local search space by phrasing states and actions as individual Multi Arm Bandit problems, and repeatedly performing Upper Confidence Bound selection of the actions which provide the best balance of exploitation and exploration. This method is also known as Upper Confidence bound applied to Trees, or UCT search.[[1]](#1)

Transposition Driven Scheduling is a load balancing technique that allows a program to efficiently scale to many threads while searching for solutions, and has historically been applied to puzzle games. It works by having the program globally agree to allocate ownership of certain gamestates to certain threads, based on the state's hash. This means that any thread can calculate who the owner is of a state that has never been seen before. 

### Work Requesting

One of the challenges specifically of searching Tetris that computing legal actions is very costly and highly non-uniform. In some states, it is a trivial computation while in others, it can take a long time. This means that existing TDS-MCTS methods such as those by Yoshizoe et al.[[3]](#3) encounter significant congestion issues when applied to this domain without dynamic load balancing.

To address this, Nana workers try to predict when they are at risk of running out of work, and pre-emptively request to borrow work from individual workers using a tuned heuristic taking into account load statistics and memory topology.

### Information State MCTS

Nana uses a special variant of MCTS that combines state abstraction with open-loop MCTS [[5]](#5) to games with randomness and hidden information. It is based on Information State MCTS [[6]](#6) which was designed for games with hidden information.

State abstraction allows Nana to combine search statistics on gamestates that look similar but differ superficially, while open-loop MCTS walks forward random simulations of the game without needing to know the underlying transition probabilities.

The result of this is that Nana can optimize against simulated random outcomes in real time, which is part of the reason it needs to search so many nodes compared to traditional engines.

### Gated Backpropagation

Nana's search parallelization is based on the optimized version of TDS-MCTS presented by Yang et al. 2020[[4]](#4), named MP-MCTS. By skipping redundant backpropagations, MP-MCTS significantly reduces root contention.

To adapt MP-MCTS's optimizations to ISMCTS, Nana resamples the environment dynamics based on the action table and hands the rollout off to another worker if states diverge. This restores the statistical properties of sequential ISMCTS.

### Markovian Opponent Modelling

To model garbage-trading dynamics and macro state, Nana uses an internal simulator that implements an Input-Output Hidden Markov Model (IOHMM). [[7]](#7) 

The IOHMM models an opponent as a probability distribution over a fixed number of states, with transitions conditioned on incoming damage.

Nana cannot see the IOHMM's internal state, and treats it as a hidden variable for ISMCTS to abstract over. This separation of search and modelling allows Nana to allocate focus between stacking and handling multi agent interactions. This is often referred to as the micro/macro split in Versus Tetris.

The default opponent model is the 8-state IOHMM trained on Nana's own play. In principle, it could be retuned or replaced with a custom model supporting the same input-output structure.

## Scaling Experiments

We benchmarked simulations per second on a 32-core AMD EPYC 9R45 (SMT disabled), at every thread count from 1 to 32 on two positions. 

The engine achieves throughput efficiency of 80% on the first 8 cores, stablizing at around 60% on 32 cores. The loss in efficiency is largely attributable to NUMA overhead. Reducing cross-chiplet communication remains an open problem.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./docs/scaling-sims-dark.png">
  <img width="800" alt="Simulations per second against threads, 1 to 32: about 80k at 1 thread, 500k at 8 and 1.54M at 32, below the dashed linear-scaling line" src="./docs/scaling-sims-light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./docs/scaling-eff-dark.png">
  <img width="800" alt="Parallel efficiency against threads: 100% at 1 thread, about 79% at 8, 72% at 9 and about 60% at 32" src="./docs/scaling-eff-light.png">
</picture>

To reproduce, run `scripts/speed_scaling.sh` on the machine (or through `scripts/run_ec2_bench.sh` with `SCRIPT=speed_scaling.sh`), then plot it with `node scripts/plot_speed.mjs <results dir>`.

### Playing Strength

We also measured end-to-end strength scaling by comparing more threads vs. more thinking time.  

Every configuration played 300 games of APM survival, and was scored by its attack per piece over the first 50 pieces (a bot that dies early sends nothing for the pieces it didn't play). 

Interestingly, over the regime we tested, a single thread given 80 to 2560 ms per move gained a consistent 6% APP per doubling of think time.

On 32 cores, Nana plays about as well as a single core given 11x thinking time.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./docs/strength-effort-dark.png">
  <img width="800" alt="Attack per piece against simulations per move: one thread thinking longer rises from 0.79 at 6k simulations to 1.08 at 165k; more threads at 80 ms track it up to 4 threads, then fall below it, reaching 0.97 at 32 threads and 119k simulations" src="./docs/strength-effort-light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./docs/strength-speedup-dark.png">
  <img width="800" alt="Speedup over one thread against threads, 1 to 32: simulations per move reach 21.5x at 32 threads, while strength speedup follows the ideal line to 4 threads and reaches 10.9x at 32" src="./docs/strength-speedup-light.png">
</picture>

To reproduce, run `scripts/strength_batches.sh` through `scripts/run_ec2_bench.sh` (e.g. `SCRIPT=strength_batches.sh scripts/run_ec2_bench.sh <host> <key> <ref> MULTS=1,2,4,8,16,32 PAR_THREADS=2,4,8,16,32`), then `python scripts/strength_report.py <batch dirs>` for the numbers and `python scripts/plot_strength.py <batch dirs>` for these plots.


## Usage

To use, Nana needs to be built from its C++ source code using CMake, i.e.

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
```

It supports a standard JSON-based interface for bots playing Tetris-like games, known as the Tetris Bot Protocol: https://github.com/tetris-bot-protocol. This version is compiled to a ``tbp`` executable.

There is also a standalone, graphical version of the bot that requires SDL2.

## References

1. https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
2. https://en.wikipedia.org/wiki/Transposition-driven_scheduling
3. Scalable Distributed Monte-Carlo Tree Search, Yoshizoe et al. 2011 (https://www.graco.c.u-tokyo.ac.jp/~kaneko/papers/socs2011pmcts.pdf)
4. Practical Massively Parallel Monte-Carlo Tree Search Applied to Molecular Design, Yang et al. 2020 (https://arxiv.org/pdf/2006.10504)
5. Open Loop Optimistic Planning, Bubeck and Munos 2010 (https://chercheurs.lille.inria.fr/munos/papers/files/olop.pdf)
6. Information Set Monte Carlo Tree Search, Cowling et al. 2012 (https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf)
7. An Input Output HMM Architecture, Bengio and Frasconi 1995 (https://www.iro.umontreal.ca/~lisa/bib/pub_subject/language/pointeurs/iohmm-nips7.pdf)