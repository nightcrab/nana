<p align="center">
<img class="only-light" height="150px" src="../logo.png"/>
</p>
<br>

## MCTS

### Abstract State MCTS

Nana uses a special variant of MCTS that combines state abstraction with open-loop MCTS [[5]](#5) to games with randomness and hidden information. It is based on Information State MCTS [[6]](#6) which was designed for games with hidden information.

State abstraction[[8]](#8) allows Nana to combine search statistics on gamestates that look similar but differ superficially, while open-loop MCTS walks forward random simulations of the game without needing to know the underlying transition probabilities.

In particular, Nana chooses to abstract the positions of garbage holes past a certain depth, as well as the future queue; as both introduce noisy speculation with little gain. Instead, boards which differ only in these respects accumulate the same search statistics on the basis that their best moves are often the same.

### PUCT-Suffix-Expectimax

To balance exploration and exploitation, Nana uses PUCT[[10]](#10); a variant of UCB that incorporates a policy prior. 

We combine this with an Expectimax-based update rule[[8]](#8) which in loose terms defines a state's reward as that of its best action, where the action's reward is averaged over chance states, sampled according to their visit distribution in the tree during search.

Since state abstraction allows paths to converge, we store leaf reward as a sequence of rewards, and only scores nodes according to their futures and not their past. We just refer to this as 'suffix credit'.

## Multithreading

Under Nana's parallelization model, the game's (abstract) state space is partitioned between all available cores with no overlap. This is a distributed computing model that assumes workers don't share memory. Therefore, all communication is message based. This eliminates typical lock contention, but comes with nontrivial engineering overhead and other considerations.

### Backpropagation Optimization (MP-MCTS)

Nana's search parallelization is based on the optimized version of TDS-MCTS presented by Yang et al. 2020[[4]](#4), named MP-MCTS. By skipping redundant backpropagations, MP-MCTS significantly reduces root contention.

To adapt the MP-MCTS optimization correctly to ISMCTS, Nana resamples the environment dynamics based on the action table, and hands off the job once the trajectory diverges. This ensures the search doesn't degrade statistically, while eliminating unnecessary communication on the shared prefix.

### Work Requesting

One of the challenges specifically of searching Tetris that computing legal actions is very costly and highly non-uniform. In some states, it is a trivial computation while in others, it can take a long time. This means that existing TDS-MCTS methods such as those by Yoshizoe et al.[[3]](#3) encounter significant congestion issues when applied to this domain without dynamic load balancing.

To address this, Nana workers try to predict when they are at risk of running out of work, and pre-emptively request to borrow work from individual workers using a tuned heuristic taking into account load statistics and memory topology.

## Two-player Evaluation

Nana uses a bilinear evaluator in its opponent's board at the root and its own during search. The evaluator is coded as a weight matrix from the opponent's features to Nana's own, and this matrix is decomposed into a sum of rank-1 matrices which are all individually expressed as the product of two vectors. Each term's gate also shifts the search's balance between attack rate and evaluation in its backed-up values.

The opponent conditioning matrix is trained over several autotuner rounds using a greedy rank-1 update method[[11]](#11). This allows each new matrix to correct for the residuals of the previous matrix. Intuitively, each sum term corresponds to a new linear strategy that depends on the opponent's boardstate.

## References

1. https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
2. https://en.wikipedia.org/wiki/Transposition-driven_scheduling
3. Scalable Distributed Monte-Carlo Tree Search, Yoshizoe et al. 2011 (https://www.graco.c.u-tokyo.ac.jp/~kaneko/papers/socs2011pmcts.pdf)
4. Practical Massively Parallel Monte-Carlo Tree Search Applied to Molecular Design, Yang et al. 2020 (https://arxiv.org/pdf/2006.10504)
5. Open Loop Optimistic Planning, Bubeck and Munos 2010 (https://chercheurs.lille.inria.fr/munos/papers/files/olop.pdf)
6. Information Set Monte Carlo Tree Search, Cowling et al. 2012 (https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf)
7. An Input Output HMM Architecture, Bengio and Frasconi 1995 (https://www.iro.umontreal.ca/~lisa/bib/pub_subject/language/pointeurs/iohmm-nips7.pdf)
8. State Aggregation in Monte Carlo Tree Search, Hostetler et al. 2014 (https://ojs.aaai.org/index.php/AAAI/article/view/9066)
9. Reinforcement Learning via AIXI Approximation, Veness et al. 2010 (https://arxiv.org/html/1007.2049v1)
10. Multi-armed bandits with episode context, Rosin 2011 (https://chrisrosin.com/isaim2010final.pdf)
11. On the Convergence of a Greedy Rank-One Update Algorithm
for a Class of Linear Systems, Amine Ammar 2010 (https://hal.science/hal-01007145/document)