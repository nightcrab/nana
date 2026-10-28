"""Fit the opponent IOHMM (cpp/Tetris/opponent/IOHMM.hpp) to logged games and write its parameters.

usage:
    python scripts/opponent/train_markov_opponent.py LOG.csv [LOG2.csv ...] [--out PATH]
    python scripts/opponent/train_markov_opponent.py --placeholder      # trivial model, to bootstrap a build
    python scripts/opponent/train_markov_opponent.py --compare A.csv B.csv  # summary statistics side by side
"""

import argparse
import copy
import csv
import re
import sqlite3
import os
import sys
from collections import defaultdict

import numpy as np

STATES = 8
ACTIONS = 6
JEOPARDY = 3
SINCE = 10
ATTACKS = 8
DEATH = ATTACKS
EMISSIONS = ATTACKS + 1
GARBAGE_CAP = 8.0

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "..", "cpp", "Tetris", "opponent", "IOHMMParams.hpp")


class Games:
    def __init__(self, rows_by_game):
        self.ids = sorted(rows_by_game)
        lengths = [len(rows_by_game[g]) for g in self.ids]
        G, T = len(self.ids), max(lengths)
        self.lengths = np.array(lengths)
        self.sent = np.zeros((G, T))
        self.attack = np.zeros((G, T))
        self.meter = np.zeros((G, T))
        self.meter_before = np.full((G, T), np.nan)
        self.dead = np.zeros((G, T), dtype=bool)
        self.mask = np.zeros((G, T), dtype=bool)
        for g, gid in enumerate(self.ids):
            rows = sorted(rows_by_game[gid], key=lambda r: r[0])
            n = len(rows)
            arr = np.array([r[1:] for r in rows], dtype=float)
            self.sent[g, :n] = arr[:, 0]
            self.attack[g, :n] = arr[:, 1]
            self.meter[g, :n] = arr[:, 3]
            self.dead[g, :n] = arr[:, 4] != 0
            self.meter_before[g, :n] = arr[:, 5]
            self.mask[g, :n] = True
        self.action = np.minimum(self.sent, ACTIONS - 1).astype(int)
        self.emission = np.where(self.dead, DEATH, np.minimum(np.round(self.attack), ATTACKS - 1)).astype(int)
        self.fresh = np.zeros((G, T), dtype=bool)
        self.fresh[:, 0] = True
        self.fresh[:, 1:] = self.dead[:, :-1]
        self.since = np.zeros((G, T), dtype=int)
        for t in range(1, T):
            self.since[:, t] = np.where(self.emission[:, t - 1] == 0, np.minimum(self.since[:, t - 1] + 1, SINCE - 1), 0)

    def subset(self, index):
        out = object.__new__(Games)
        for name, value in vars(self).items():
            if isinstance(value, np.ndarray):
                setattr(out, name, value[index])
            else:
                setattr(out, name, [value[i] for i in index])
        return out

    def steps(self):
        return int(self.mask.sum())


def load(paths):
    rows_by_game = defaultdict(list)
    for file_index, path in enumerate(paths):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                gid = (file_index, int(row["game"]))
                rows_by_game[gid].append((int(row["piece"]), float(row["sent"]), float(row["opp_attack"]),
                                          float(row["opp_sent"]), float(row["opp_meter"]), int(row["opp_dead"]),
                                          np.nan))
    return Games(rows_by_game)


def load_stadium(path, file_index, rows_by_game, players):
    db = sqlite3.connect(path)
    rows_by_id = defaultdict(list)
    for row in db.execute("SELECT game_id, move_index, state, p1_attack, p1_meter, p2_attack, p2_meter FROM Data "
                          "ORDER BY game_id, move_index"):
        rows_by_id[row[0]].append(row)
    for game_id, rows in rows_by_id.items():
        playing = [r for r in rows if r[2] == "PLAYING"]
        final = [r for r in rows if r[2] != "PLAYING"]
        if not playing or len(final) != 1:
            continue
        result = final[0][2]
        for player in players:
            attack_col, meter_col = (3, 4) if player == 1 else (5, 6)
            other_col = 5 if player == 1 else 3
            lost = result == "DRAW" or result == ("P2_WIN" if player == 1 else "P1_WIN")
            gid = (file_index, game_id, player)
            for t, r in enumerate(playing):
                sent = playing[t - 1][other_col] if t > 0 else 0
                rows_by_game[gid].append((t, sent, r[attack_col], 0.0, np.nan, 0, r[meter_col]))
            if lost:
                t = len(playing)
                rows_by_game[gid].append((t, playing[-1][other_col], 0.0, 0.0, np.nan, 1, final[0][meter_col]))


def load_any(paths, players=(1, 2)):
    csvs = [p for p in paths if not p.endswith(".db")]
    games = load(csvs) if csvs else None
    rows_by_game = defaultdict(list)
    if games is not None:
        for g, gid in enumerate(games.ids):
            for t in range(games.lengths[g]):
                rows_by_game[("csv",) + gid].append((t, games.sent[g, t], games.attack[g, t], 0.0, games.meter[g, t],
                                                     int(games.dead[g, t]), np.nan))
    for file_index, path in enumerate(paths):
        if path.endswith(".db"):
            load_stadium(path, file_index, rows_by_game, players)
    return Games(rows_by_game)


def replay_meter(games):
    G, T = games.mask.shape
    meter = np.zeros(G)
    before = np.zeros((G, T))
    after = np.zeros((G, T))
    for t in range(T):
        meter = meter + games.sent[:, t]
        before[:, t] = meter
        alive = ~games.dead[:, t]
        attack = games.attack[:, t]
        cancelled = np.where(alive & (attack > 0), np.minimum(attack, meter), 0)
        tanked = np.where(alive & (attack <= 0), np.minimum(meter, GARBAGE_CAP), 0)
        meter = meter - cancelled - tanked
        after[:, t] = meter
    return before, after


def bucket(jeopardy, limits):
    b = np.zeros(jeopardy.shape, dtype=int)
    for limit in limits:
        b += jeopardy >= limit
    return b


def choose_jeopardy(games, meter, max_limit=30, min_share=0.005):
    assert JEOPARDY == 3, "choose_jeopardy searches two limits"
    E = EMISSIONS
    p = np.minimum(np.floor(meter[games.mask]), max_limit).astype(int)
    e = games.emission[games.mask]
    hist = np.zeros((max_limit + 1, E))
    np.add.at(hist, (p, e), 1)
    cum = np.vstack([np.zeros(E), np.cumsum(hist, axis=0)])
    n = hist.sum()

    def ll(lo, hi):
        counts = cum[hi] - cum[lo]
        total = counts.sum()
        if total < min_share * n:
            return None
        probs = (counts + 0.5) / (total + 0.5 * E)
        return float((counts * np.log(probs)).sum())

    best = None
    for l1 in range(1, max_limit):
        for l2 in range(l1 + 1, max_limit + 1):
            parts = [ll(0, l1), ll(l1, l2), ll(l2, max_limit + 1)]
            if None in parts:
                continue
            score = sum(parts)
            if best is None or score > best[0]:
                best = (score, (float(l1), float(l2)))
    return best


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


RUNGS = 4


def rung_offsets():
    rung = np.arange(STATES) % RUNGS
    return rung[None, :] - rung[:, None] + RUNGS - 1


class Model:
    def __init__(self, rng, emission_prior, cycle=False, grid=False, full=False):
        self.initial = np.full(STATES, 1.0 / STATES)
        self.U = np.zeros((ACTIONS, STATES))
        self.V = np.zeros((JEOPARDY, STATES))
        self.X = np.zeros((SINCE, STATES))
        self.grid = grid
        self.full = full
        self.Y = np.zeros((ACTIONS, STATES, STATES))
        self.K = np.zeros((ACTIONS, 2 * RUNGS - 1))
        if grid:
            assert STATES % RUNGS == 0, "the grid needs STATES to be a multiple of RUNGS"
            mode, rung = np.divmod(np.arange(STATES), RUNGS)
            same_mode = mode[:, None] == mode[None, :]
            d = rung[None, :] - rung[:, None]
            base = (0.55 * (same_mode & (d == 0)) + 0.2 * (same_mode & (d == -1))
                    + 0.1 * (~same_mode & (d == 0)) + 0.01)
            self.W = np.log(base) + 0.1 * rng.standard_normal((STATES, STATES))
            shift = np.array([0.0, 0.3, 0.6, 1.0, 1.5, 2.0])[:ACTIONS]
            offsets = np.arange(2 * RUNGS - 1) - (RUNGS - 1)
            self.K = -0.5 * (offsets[None, :] - shift[:, None]) ** 2
            rows = []
            for z in range(STATES):
                r = emission_prior.copy()
                r[DEATH] *= [0.1, 0.5, 2.0, 6.0][rung[z]] if RUNGS == 4 else (1 + rung[z])
                r[1:DEATH] *= 2.0 if mode[z] else 0.5
                rows.append(r / r.sum())
            self.emission = np.stack([rng.dirichlet(r * 50 + 0.1) for r in rows])
        elif cycle:
            stay, step = np.eye(STATES), np.roll(np.eye(STATES), 1, axis=1)
            self.W = np.log(0.45 * stay + 0.5 * step + 0.05 / STATES) + 0.05 * rng.standard_normal((STATES, STATES))
            quiet = emission_prior.copy()
            quiet[1:DEATH] *= 0.1
            loud = emission_prior.copy()
            loud[0] *= 0.2
            rows = [quiet] * (STATES - 1) + [loud]
            self.emission = np.stack([rng.dirichlet(r / r.sum() * 50 + 0.1) for r in rows])
        else:
            self.W = np.log(np.full((STATES, STATES), 0.2 / STATES) + 0.8 * np.eye(STATES)) + 0.1 * rng.standard_normal((STATES, STATES))
            self.emission = np.stack([rng.dirichlet(emission_prior * 20 + 0.1) for _ in range(STATES)])

    def transition(self):
        logits = (self.W[None, None, None] + self.U[:, None, None, None, :] + self.V[None, :, None, None, :]
                  + self.X[None, None, :, None, :])
        if self.grid:
            logits = logits + self.K[:, rung_offsets()][:, None, None, :, :]
        if self.full:
            logits = logits + self.Y[:, None, None, :, :]
        return softmax(logits)


def forward_backward(model, games, jeopardy_bucket, return_gamma=False):
    G, T = games.mask.shape
    A = model.transition()
    B = model.emission
    lik = B[:, games.emission].transpose(1, 2, 0)
    lik = np.where(games.mask[..., None], lik, 1.0)

    alpha = np.zeros((G, T, STATES))
    scale = np.ones((G, T))
    prev = None
    for t in range(T):
        if t == 0:
            prior = np.broadcast_to(model.initial, (G, STATES))
        else:
            At = A[games.action[:, t], jeopardy_bucket[:, t], games.since[:, t]]
            prior = np.einsum("gi,gij->gj", prev, At)
            prior = np.where(games.fresh[:, t, None], model.initial[None], prior)
        a = prior * lik[:, t]
        s = a.sum(axis=1)
        s = np.where(games.mask[:, t], s, 1.0)
        a = a / s[:, None]
        scale[:, t] = s
        alpha[:, t] = a
        prev = a

    beta = np.ones((G, T, STATES))
    for t in range(T - 2, -1, -1):
        At1 = A[games.action[:, t + 1], jeopardy_bucket[:, t + 1], games.since[:, t + 1]]
        nb = np.einsum("gij,gj->gi", At1, lik[:, t + 1] * beta[:, t + 1]) / scale[:, t + 1, None]
        cut = games.fresh[:, t + 1] | ~games.mask[:, t + 1]
        beta[:, t] = np.where(cut[:, None], 1.0, nb)

    gamma = alpha * beta
    gamma /= gamma.sum(axis=2, keepdims=True)
    gamma *= games.mask[..., None]

    init_counts = (gamma * games.fresh[..., None]).sum(axis=(0, 1))
    em_counts = np.zeros((STATES, EMISSIONS))
    for e in range(EMISSIONS):
        em_counts[:, e] = (gamma * (games.emission == e)[..., None]).sum(axis=(0, 1))

    tr_counts = np.zeros((ACTIONS, JEOPARDY, SINCE, STATES, STATES))
    for t in range(1, T):
        use = games.mask[:, t] & ~games.fresh[:, t]
        if not use.any():
            continue
        At = A[games.action[use, t], jeopardy_bucket[use, t], games.since[use, t]]
        xi = alpha[use, t - 1][:, :, None] * At * (lik[use, t] * beta[use, t])[:, None, :]
        xi /= xi.sum(axis=(1, 2), keepdims=True)
        np.add.at(tr_counts, (games.action[use, t], jeopardy_bucket[use, t], games.since[use, t]), xi)

    loglik = float(np.log(scale[games.mask]).sum())
    if return_gamma:
        return loglik, init_counts, tr_counts, em_counts, gamma
    return loglik, init_counts, tr_counts, em_counts


def death_value(model, games, jeopardy_bucket, horizon=10, discount=None, pseudo=5.0, max_truncation=0.05):
    *_, gamma = forward_backward(model, games, jeopardy_bucket, return_gamma=True)
    G, T = games.mask.shape
    t_index = np.arange(T)[None, :]
    dead = games.dead & games.mask
    never = 10 ** 6
    until = np.full((G, T), never)
    for t in range(T - 2, -1, -1):
        until[:, t] = np.where(dead[:, t + 1], 1, np.where(until[:, t + 1] < never, until[:, t + 1] + 1, never))
    left = games.lengths[:, None] - 1 - t_index
    if discount is None:
        label = (until <= horizon).astype(float)
        known = (until <= horizon) | (left >= horizon)
    else:
        label = np.where(until < never, discount ** np.minimum(until, 10 ** 4).astype(float), 0.0)
        known = (until < never) | (discount ** np.maximum(left, 0).astype(float) <= max_truncation)
    use = games.mask & ~games.dead & known & (left >= 1)
    next_j = np.zeros_like(jeopardy_bucket)
    next_j[:, :-1] = jeopardy_bucket[:, 1:]
    next_s = np.zeros_like(games.since)
    next_s[:, :-1] = games.since[:, 1:]
    num = np.zeros((JEOPARDY, SINCE, STATES))
    den = np.zeros((JEOPARDY, SINCE, STATES))
    w = gamma[use]
    np.add.at(den, (next_j[use], next_s[use]), w)
    np.add.at(num, (next_j[use], next_s[use]), w * label[use][:, None])
    pooled = num.sum(axis=(0, 1)) / np.maximum(den.sum(axis=(0, 1)), 1e-9)
    table = (num + pseudo * pooled) / (den + pseudo)
    return table.transpose(2, 0, 1), den.transpose(2, 0, 1)


class LoadedModel:
    def __init__(self, path):
        numbers = [float(x) for x in re.findall(r"(-?\d+(?:\.\d*)?(?:e[-+]?\d+)?)f\b", open(path).read())]
        sizes = [STATES, ACTIONS * JEOPARDY * SINCE * STATES * STATES, STATES * EMISSIONS, ATTACKS, JEOPARDY - 1]
        if len(numbers) < sum(sizes):
            sys.exit(f"{path} doesn't hold a model with these dimensions")
        parts = []
        for size in sizes:
            parts.append(np.array(numbers[:size]))
            numbers = numbers[size:]
        self.initial = parts[0] / parts[0].sum()
        transition = parts[1].reshape(ACTIONS, JEOPARDY, SINCE, STATES, STATES)
        self._transition = transition / transition.sum(axis=-1, keepdims=True)
        emission = parts[2].reshape(STATES, EMISSIONS)
        self.emission = emission / emission.sum(axis=-1, keepdims=True)
        self.attack_value = parts[3]
        self.limits = tuple(parts[4])
        self.stadium_rules = bool(re.search(r"// stadium_rules\s+true", open(path).read()))

    def transition(self):
        return self._transition


def m_step(model, init_counts, tr_counts, em_counts, l2=1.0, steps=60, lr=0.05):
    model.initial = (init_counts + 1e-3) / (init_counts + 1e-3).sum()
    model.emission = (em_counts + 1e-4) / (em_counts + 1e-4).sum(axis=1, keepdims=True)

    total = tr_counts.sum()
    n_from = tr_counts.sum(axis=4, keepdims=True)
    if model.grid:
        params = [model.W, model.K, model.V, model.X]
        offset_onehot = np.eye(2 * RUNGS - 1)[rung_offsets()]
    else:
        params = [model.W, model.U, model.V, model.X]
    if model.full:
        params.append(model.Y)
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    for k in range(1, steps + 1):
        G = (tr_counts - n_from * model.transition()) / total
        if model.grid:
            action_grad = np.einsum("aij,ijo->ao", G.sum(axis=(1, 2)), offset_onehot) - 2 * l2 * model.K / total
        else:
            action_grad = G.sum(axis=(1, 2, 3)) - 2 * l2 * model.U / total
        grads = [G.sum(axis=(0, 1, 2)),
                 action_grad,
                 G.sum(axis=(0, 2, 3)) - 2 * l2 * model.V / total,
                 G.sum(axis=(0, 1, 3)) - 2 * l2 * model.X / total]
        if model.full:
            grads.append(G.sum(axis=(1, 2)) - 2 * l2 * model.Y / total)
        for i, (p, g) in enumerate(zip(params, grads)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p += lr * (m[i] / (1 - 0.9 ** k)) / (np.sqrt(v[i] / (1 - 0.999 ** k)) + 1e-8)


def fit(games, jeopardy_bucket, rng, restarts=4, warmup=15, iterations=1000, tol=1e-6, patience=10, l2=1.0, topology="free",
        log=print):
    prior = np.bincount(games.emission[games.mask], minlength=EMISSIONS).astype(float)
    prior /= prior.sum()
    n = games.steps()

    candidates = []
    for r in range(restarts):
        model = Model(rng, prior, cycle=r % 2 == 0, grid=topology == "grid", full=topology == "full")
        for _ in range(warmup):
            ll, *counts = forward_backward(model, games, jeopardy_bucket)
            m_step(model, *counts, l2=l2)
        candidates.append((ll, model))
        log(f"  restart {r}: {ll / n:.5f} nats/piece after {warmup} iterations")
    ll, model = max(candidates, key=lambda c: c[0])

    best_ll, best = -np.inf, None
    stale = 0
    for it in range(iterations):
        ll, *counts = forward_backward(model, games, jeopardy_bucket)
        if it % 10 == 0:
            log(f"  iteration {it}: {ll / n:.5f} nats/piece")
        if ll > best_ll + tol * n:
            stale = 0
        else:
            stale += 1
        if ll > best_ll:
            best_ll, best = ll, copy.deepcopy(model)
        if stale >= patience:
            break
        m_step(model, *counts, l2=l2)
    log(f"  converged after {it + 1} iterations: {best_ll / n:.5f} nats/piece")
    return best


def baseline_loglik(train, test, test_bucket, train_bucket):
    counts = np.zeros((JEOPARDY, EMISSIONS))
    np.add.at(counts, (train_bucket[train.mask], train.emission[train.mask]), 1)
    probs = (counts + 0.5) / (counts + 0.5).sum(axis=1, keepdims=True)
    return float(np.log(probs[test_bucket[test.mask], test.emission[test.mask]]).sum())


def simulate(model, attack_value, limits, games, rng):
    A = model.transition()
    G, T = games.mask.shape
    attack = np.zeros((G, T))
    dead = np.zeros((G, T), dtype=bool)
    for g in range(G):
        meter = 0.0
        fresh = True
        z = 0
        since = 0
        for t in range(games.lengths[g]):
            if t > 0 and dead[g, t - 1]:
                fresh = True
            meter += games.sent[g, t]
            logged = games.meter_before[g, t]
            c = int(sum((meter if np.isnan(logged) else logged) >= l for l in limits))
            a = games.action[g, t]
            z = rng.choice(STATES, p=model.initial) if fresh else rng.choice(STATES, p=A[a, c, since, z])
            fresh = False
            e = rng.choice(EMISSIONS, p=model.emission[z])
            since = min(since + 1, SINCE - 1) if e == 0 else 0
            if e == DEATH:
                dead[g, t] = True
                continue
            x = attack_value[e]
            attack[g, t] = x
            if x > 0:
                meter -= min(x, meter)
            else:
                meter -= min(meter, GARBAGE_CAP)
    return attack, dead


def summary(sent, attack, dead, mask):
    a = attack[mask]
    out = {}
    out["pieces"] = int(mask.sum())
    out["attack/piece"] = a.mean()
    out["P(attack>0)"] = (a > 0).mean()
    for v in [1, 2, 3, 4, 5]:
        out[f"P(attack={v})"] = (np.round(a) == v).mean()
    out["P(attack>=6)"] = (a >= 6).mean()
    out["deaths/1000"] = 1000 * dead[mask].mean()
    on = (attack > 0) & mask
    for k in [1, 2, 3, 5, 8]:
        both = (on[:, :-k] & on[:, k:]).sum()
        out[f"P(on@+{k}|on)"] = both / max(on[:, :-k].sum(), 1)
    T = attack.shape[1]
    win = np.zeros_like(attack)
    for k in range(1, 6):
        win[:, :T - k] += np.where(mask[:, k:], attack[:, k:], 0)
    full = np.zeros_like(mask)
    full[:, :T - 5] = mask[:, 5:]
    big = (sent >= 4) & full
    none = (sent == 0) & full
    out["attack next5 | sent>=4"] = win[big].mean() / 5 if big.any() else float("nan")
    out["attack next5 | sent=0"] = win[none].mean() / 5 if none.any() else float("nan")
    return out


def print_table(columns):
    names = list(columns)
    keys = list(columns[names[0]])
    print(f"{'':28s}" + "".join(f"{n:>14s}" for n in names))
    for k in keys:
        print(f"{k:28s}" + "".join(f"{columns[n][k]:14.4f}" for n in names))


def fmt(x):
    text = f"{x:.7g}"
    if "." not in text and "e" not in text:
        text += ".0"
    return text + "f"


def braces(values):
    return "{{" + ", ".join(values) + "}}"


def write_header(path, initial, transition, emission, attack_value, limits, value_table, horizon, discount, source,
                 stadium_rules=False):
    lines = []
    lines.append(f"// generated by scripts/opponent/train_markov_opponent.py from {source}; don't edit")
    lines.append("#pragma once")
    lines.append('#include "IOHMM.hpp"')
    lines.append("")
    lines.append("namespace opponent {")
    lines.append(f"static_assert(STATES == {STATES} && ACTIONS == {ACTIONS} && JEOPARDY == {JEOPARDY} && SINCE == {SINCE} && "
                 f"ATTACKS == {ATTACKS}, \"IOHMMParams.hpp was fitted with other dimensions\");")
    lines.append("")
    lines.append("inline constexpr IOHMMParams fitted_iohmm = {")
    lines.append("    // initial")
    lines.append("    " + braces(fmt(x) for x in initial) + ",")
    lines.append("    // transition[action][jeopardy][since][from][to]")
    action_blocks = []
    for a in range(ACTIONS):
        jeopardy_blocks = []
        for c in range(JEOPARDY):
            since_blocks = []
            for d in range(SINCE):
                rows = ",\n".join("                " + braces(fmt(x) for x in transition[a, c, d, i]) for i in range(STATES))
                since_blocks.append("            {{\n" + rows + "\n            }}")
            jeopardy_blocks.append("        {{\n" + ",\n".join(since_blocks) + "\n        }}")
        action_blocks.append(f"    // action {a}\n" + "    {{\n" + ",\n".join(jeopardy_blocks) + "\n    }}")
    lines.append("    {{\n" + ",\n".join(action_blocks) + "\n    }},")
    lines.append("    // emission[state][attack bin, death]")
    rows = ",\n".join("        " + braces(fmt(x) for x in emission[i]) for i in range(STATES))
    lines.append("    {{\n" + rows + "\n    }},")
    lines.append("    // attack_value")
    lines.append("    " + braces(fmt(x) for x in attack_value) + ",")
    lines.append("    // jeopardy_limits")
    lines.append("    " + braces(fmt(x) for x in limits) + ",")
    what = f"E[{discount}^(pieces until death)]" if discount else f"P(dying within the next {horizon} pieces)"
    lines.append(f"    // death_value[state][jeopardy][since]: {what}")
    blocks = []
    for z in range(STATES):
        rows = ",\n".join("        " + braces(fmt(x) for x in value_table[z, c]) for c in range(JEOPARDY))
        blocks.append("    {{\n" + rows + "\n    }}")
    lines.append("    {{\n" + ",\n".join(blocks) + "\n    }},")
    lines.append("    // horizon (0 when discounted), discount (0 for a fixed horizon)")
    lines.append(f"    {0 if discount else horizon},")
    lines.append("    " + fmt(discount or 0.0) + ",")
    lines.append("    // stadium_rules")
    lines.append("    " + ("true" if stadium_rules else "false") + ",")
    lines.append("};")
    lines.append("")
    lines.append("}")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def write_placeholder(path):
    initial = np.eye(STATES)[0]
    transition = np.broadcast_to(np.eye(STATES), (ACTIONS, JEOPARDY, SINCE, STATES, STATES))
    emission = np.zeros((STATES, EMISSIONS))
    emission[:, 0] = 1
    attack_value = np.arange(ATTACKS, dtype=float)
    write_header(path, initial, transition, emission, attack_value, np.arange(1, JEOPARDY) * 4.0,
                 np.zeros((STATES, JEOPARDY, SINCE)), 1, None, "nothing (placeholder: never attacks)")


def report_death_value(model, games, jeopardy_bucket, args):
    table, weight = death_value(model, games, jeopardy_bucket, args.horizon, args.discount)
    what = f"E[{args.discount}^(pieces until death)]" if args.discount else f"P(dying within {args.horizon} pieces)"
    print(f"{what} by state (pooled over jeopardy and since), and the states' weight:")
    pooled = (table * weight).sum(axis=(1, 2)) / np.maximum(weight.sum(axis=(1, 2)), 1e-9)
    share = weight.sum(axis=(1, 2)) / weight.sum()
    print("  " + "  ".join(f"z{z}: {pooled[z]:.3f} ({share[z]:.2f})" for z in range(STATES)))
    by_j = (table * weight).sum(axis=(0, 2)) / np.maximum(weight.sum(axis=(0, 2)), 1e-9)
    print("  by jeopardy bucket: " + ", ".join(f"{x:.3f}" for x in by_j))
    by_s = (table * weight).sum(axis=(0, 1)) / np.maximum(weight.sum(axis=(0, 1)), 1e-9)
    print("  by pieces since its last attack: " + ", ".join(f"{x:.3f}" for x in by_s))
    return table


def refit_death_value(args):
    model = LoadedModel(args.from_header)
    games = load_any(args.logs, tuple(int(p) for p in args.players.split(",")))
    meter, _ = replay_meter(games)
    meter = np.where(~np.isnan(games.meter_before), games.meter_before, meter)
    jeopardy_bucket = bucket(meter, model.limits)
    table = report_death_value(model, games, jeopardy_bucket, args)
    source = ", ".join(os.path.basename(p) for p in args.logs) + f" (model from {os.path.basename(args.from_header)})"
    stadium_rules = model.stadium_rules or any(p.endswith(".db") for p in args.logs)
    write_header(args.out, model.initial, model.transition(), model.emission, model.attack_value, model.limits,
                 table, args.horizon, args.discount, source, stadium_rules)
    print(f"wrote {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="*")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--placeholder", action="store_true")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    parser.add_argument("--test-share", type=float, default=0.2)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min-jeopardy-share", type=float, default=0.005,
                        help="smallest share of pieces a jeopardy bucket may hold")
    parser.add_argument("--horizon", type=int, default=10,
                        help="death_value is P(dying within this many pieces), unless --discount is given")
    parser.add_argument("--discount", type=float, default=None,
                        help="death_value is E[discount^(pieces until death)] instead")
    parser.add_argument("--from-header", metavar="PATH",
                        help="don't fit: read the model from a generated header and recompute death_value")
    parser.add_argument("--topology", choices=["free", "grid", "full"], default="free",
                        help=f"free: our attack's effect only depends on the next state; grid: states are {RUNGS} "
                             "burial rungs x modes, and our attack shifts the rung; full: free plus an "
                             "action x from x to interaction")
    parser.add_argument("--players", default="1,2", help="stadium players to model, e.g. 2 or 1,2")
    parser.add_argument("--states", type=int, default=STATES,
                        help="hidden states; IOHMM.hpp's STATES must match to use the output")
    args = parser.parse_args()
    globals()["STATES"] = args.states

    if args.placeholder:
        write_placeholder(args.out)
        print(f"wrote {args.out}")
        return

    if args.compare:
        columns = {}
        for path in args.compare:
            g = load_any([path])
            columns[os.path.basename(path)] = summary(g.sent, g.attack, g.dead, g.mask)
        print_table(columns)
        return

    if not args.logs:
        parser.error("no logs")

    if args.from_header:
        refit_death_value(args)
        return

    rng = np.random.default_rng(args.seed)
    games = load_any(args.logs, tuple(int(p) for p in args.players.split(",")))
    print(f"{len(games.ids)} games, {games.steps()} pieces, {int(games.dead[games.mask].sum())} opponent deaths")

    meter, meter_after = replay_meter(games)
    logged = ~np.isnan(games.meter_before)
    replayed = games.mask & ~logged
    if replayed.any():
        err = np.abs(meter_after - games.meter)[replayed].max()
        print(f"meter replay: max error {err:.3g}")
        if err > 1e-3:
            print("warning: the replayed meter doesn't match the log", file=sys.stderr)
    meter = np.where(logged, games.meter_before, meter)

    best = choose_jeopardy(games, meter, min_share=args.min_jeopardy_share)
    if best is None:
        sys.exit("no jeopardy limits leave every bucket --min-jeopardy-share of the pieces; lower it")
    score, limits = best
    print(f"jeopardy limits {limits} ({score / games.steps():.5f} nats/piece without hidden state)")
    jeopardy_bucket = bucket(meter, limits)
    shares = np.bincount(jeopardy_bucket[games.mask], minlength=JEOPARDY) / games.steps()
    print("jeopardy bucket shares: " + ", ".join(f"{s:.3f}" for s in shares))
    print("action shares: " + ", ".join(f"{s:.3f}" for s in np.bincount(games.action[games.mask], minlength=ACTIONS) / games.steps()))

    order = rng.permutation(len(games.ids))
    n_test = int(round(args.test_share * len(order)))
    if n_test > 0 and len(order) - n_test > 0:
        test_idx, train_idx = np.sort(order[:n_test]), np.sort(order[n_test:])
        train, test = games.subset(train_idx), games.subset(test_idx)
        print(f"fitting on {len(train_idx)} games, testing on {len(test_idx)}")
        model = fit(train, jeopardy_bucket[train_idx], rng, restarts=args.restarts, l2=args.l2,
                    topology=args.topology)
        test_ll, *_ = forward_backward(model, test, jeopardy_bucket[test_idx])
        base = baseline_loglik(train, test, jeopardy_bucket[test_idx], jeopardy_bucket[train_idx])
        print(f"held out: {test_ll / test.steps():.5f} nats/piece (jeopardy-only baseline {base / test.steps():.5f})")

    print("fitting on all games")
    model = fit(games, jeopardy_bucket, rng, restarts=args.restarts, l2=args.l2, topology=args.topology)

    attack_value = np.arange(ATTACKS, dtype=float)
    alive = games.mask & ~games.dead
    for e in range(ATTACKS):
        sel = alive & (games.emission == e)
        if sel.any():
            attack_value[e] = games.attack[sel].mean()

    sim_attack, sim_dead = simulate(model, attack_value, limits, games, rng)
    print_table({"log": summary(games.sent, games.attack, games.dead, games.mask),
                 "fitted": summary(games.sent, sim_attack, sim_dead, games.mask)})

    table = report_death_value(model, games, jeopardy_bucket, args)
    source = ", ".join(os.path.basename(p) for p in args.logs)
    stadium = [p.endswith(".db") for p in args.logs]
    if any(stadium) and not all(stadium):
        print("warning: mixing stadium games with nana_emulate logs; the model will use stadium rules", file=sys.stderr)
    write_header(args.out, model.initial, model.transition(), model.emission, attack_value, limits,
                 table, args.horizon, args.discount, source, any(stadium))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
