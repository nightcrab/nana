"""Particle swarm optimisation for a noisy fitness, in the unit cube.

The fitness is a mean over seeded games, so two evaluations of one point differ. Plain PSO keeps
each particle's best (pbest) at the score of its luckiest evaluation, and the swarm then chases
luck. Here, every iteration plays one fresh block of seeds and evaluates on it both each particle's
current position and its pbest:
  - the pbest's score is a running mean over all its evaluations, so luck averages out;
  - a position replaces its pbest only if it beats it on the same seeds (a paired comparison, which
    cancels most of the noise from the pieces and garbage);
  - the global best is the pbest with the best running mean among those evaluated at least twice.
Costs about twice the games of plain PSO per iteration; much more robust.
"""

import numpy as np


class PSO:
    def __init__(self, dim, particles, rng_seed=0, inertia=0.6, cognitive=1.5, social=1.5, vmax=0.2):
        self.dim = dim
        self.n = particles
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.vmax = vmax
        self.rng = np.random.default_rng(rng_seed)
        self.x = np.zeros((particles, dim))
        self.v = np.zeros((particles, dim))
        self.p = np.zeros((particles, dim))
        self.p_mean = np.zeros(particles)
        self.p_count = np.zeros(particles, dtype=int)
        self.iteration = 0

    # ---- set up

    def scatter(self, center, spread, keep_center=True):
        """Particles around center (unit cube) with per-dimension spread; the first one exactly at
        center, so a warm start never loses its starting point."""
        center = np.asarray(center, dtype=float)
        self.x = np.clip(center + spread * self.rng.standard_normal((self.n, self.dim)), 0, 1)
        if keep_center:
            self.x[0] = center
        self.v = 0.1 * spread * self.rng.standard_normal((self.n, self.dim))
        self.p = self.x.copy()
        self.p_mean[:] = 0
        self.p_count[:] = 0

    def grow(self, new_dim, new_center):
        """Add dimensions (--add-params): existing coordinates stay, new ones start at new_center."""
        extra = new_dim - self.dim
        fill = np.tile(np.asarray(new_center[self.dim:], dtype=float), (self.n, 1))
        self.x = np.hstack([self.x, fill])
        self.p = np.hstack([self.p, fill])
        self.v = np.hstack([self.v, np.zeros((self.n, extra))])
        self.dim = new_dim

    # ---- one iteration: ask for points, then tell their scores on a shared block of seeds

    def ask(self):
        """Points to evaluate this iteration: ('x', i) for every particle, ('p', i) for every pbest
        that has been evaluated before (a pbest that's never been scored is its particle's x)."""
        points = [("x", i, self.x[i].copy()) for i in range(self.n)]
        points += [("p", i, self.p[i].copy()) for i in range(self.n) if self.p_count[i] > 0]
        return points

    def tell(self, scores):
        """scores: {('x'|'p', i): mean fitness on this iteration's seeds}."""
        for i in range(self.n):
            fx = scores[("x", i)]
            if self.p_count[i] == 0:
                self.p[i] = self.x[i]
                self.p_mean[i] = fx
                self.p_count[i] = 1
                continue
            fp = scores[("p", i)]
            self.p_mean[i] = (self.p_mean[i] * self.p_count[i] + fp) / (self.p_count[i] + 1)
            self.p_count[i] += 1
            if fx > fp:
                # beat it on the same seeds: take its place, starting a new running mean
                self.p[i] = self.x[i]
                self.p_mean[i] = fx
                self.p_count[i] = 1
        self._move()
        self.iteration += 1

    def best(self):
        """(index, unit point, running mean, evaluations) of the global best."""
        trusted = self.p_count >= 2
        pool = np.where(trusted)[0] if trusted.any() else np.where(self.p_count >= 1)[0]
        if len(pool) == 0:
            return None
        i = pool[np.argmax(self.p_mean[pool])]
        return int(i), self.p[i].copy(), float(self.p_mean[i]), int(self.p_count[i])

    def _move(self):
        best = self.best()
        g = best[1] if best else self.x[0]
        r1 = self.rng.random((self.n, self.dim))
        r2 = self.rng.random((self.n, self.dim))
        self.v = (self.inertia * self.v + self.cognitive * r1 * (self.p - self.x)
                  + self.social * r2 * (g[None, :] - self.x))
        self.v = np.clip(self.v, -self.vmax, self.vmax)
        self.x = self.x + self.v
        # stop at the walls rather than bouncing off them
        hit = (self.x < 0) | (self.x > 1)
        self.x = np.clip(self.x, 0, 1)
        self.v[hit] = 0

    # ---- monitors

    def diversity(self):
        """mean distance of the particles from their centroid, in unit-cube units per dimension"""
        return float(np.mean(np.linalg.norm(self.x - self.x.mean(axis=0), axis=1)) / np.sqrt(self.dim))

    def at_bounds(self):
        """per dimension, the share of pbests sitting on a wall"""
        return np.mean((self.p <= 1e-6) | (self.p >= 1 - 1e-6), axis=0)

    # ---- checkpoints

    def state(self):
        return {"dim": self.dim, "n": self.n, "inertia": self.inertia, "cognitive": self.cognitive,
                "social": self.social, "vmax": self.vmax, "iteration": self.iteration,
                "x": self.x.tolist(), "v": self.v.tolist(), "p": self.p.tolist(),
                "p_mean": self.p_mean.tolist(), "p_count": self.p_count.tolist(),
                "rng": self.rng.bit_generator.state}

    @staticmethod
    def from_state(s):
        pso = PSO(s["dim"], s["n"], 0, s["inertia"], s["cognitive"], s["social"], s["vmax"])
        pso.iteration = s["iteration"]
        pso.x = np.array(s["x"])
        pso.v = np.array(s["v"])
        pso.p = np.array(s["p"])
        pso.p_mean = np.array(s["p_mean"])
        pso.p_count = np.array(s["p_count"], dtype=int)
        pso.rng.bit_generator.state = s["rng"]
        return pso
