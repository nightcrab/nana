"""Fit the opponent HMM and a board encoder together, on the target bot's own games.

usage: python scripts/train_encoder_hmm.py STADIUM.db [...] --tool PATH/nana_opp_features [--out DIR]

The search uses the opponent model in one way: a belief over hidden states at the root, then the HMM run
forward with no boards. This script trains for exactly that. At every position of the target bot the root
belief is the damage filter's prior times the encoder's evidence from the position (nana_opp_features:
the CC eval's board terms and a greedy Cold Clear rollout over the known queue), and the loss is the
log-likelihood of the bot's next HORIZON attacks under the HMM started from that belief.

The target player in each database is read from the adjacent sides.txt (the side named cc-tbp), or set
with --player. Held-out games report bits per piece for:
  pooled       unconditional attack frequencies (training games)
  shipped      MarkovOpponentParams.hpp, raw and with the search's default sharpening
  refit        the HMM refitted by EM on the target's training games, damage filter only
  two-stage    refit plus an encoder fitted to the refit's smoothed state posteriors
  joint        HMM and encoder trained together on the horizon loss, from the refit
"""

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import train_markov_opponent as m

S, A_, J, D, E = m.STATES, m.ACTIONS, m.JEOPARDY, m.SINCE, m.EMISSIONS
LOG2 = np.log(2.0)
SHIPPED = os.path.join(os.path.dirname(__file__), "..", "cpp", "Tetris", "game", "MarkovOpponentParams.hpp")


def target_player(path, forced):
    if forced:
        return forced
    sides = os.path.join(os.path.dirname(path), "sides.txt")
    if not os.path.exists(sides):
        sys.exit(f"{path}: no sides.txt beside it; pass --player")
    words = open(sides).read().split()
    p1 = words[1]
    return 1 if p1.startswith("cc") else 2


def load(paths, forced):
    rows_by_game = defaultdict(list)
    lines = {}
    for fi, path in enumerate(paths):
        player = target_player(path, forced)
        m.load_stadium(path, fi, rows_by_game, (player,))
        pre = f"p{player}_"
        cols = ["board", "current_piece", "hold", "queue_0", "queue_1", "queue_2", "queue_3", "queue_4", "combo",
                "b2b", "meter"]
        db = sqlite3.connect(path)
        by_game = defaultdict(list)
        for row in db.execute("SELECT game_id, move_index, state, " + ", ".join(pre + c for c in cols) +
                              " FROM Data ORDER BY game_id, move_index"):
            by_game[row[0]].append(row)
        for game_id, rows in by_game.items():
            playing = [r for r in rows if r[2] == "PLAYING"]
            final = [r for r in rows if r[2] != "PLAYING"]
            if not playing or len(final) != 1:
                continue
            for t, r in enumerate(playing + final):
                piece = lambda x: x if x in ("I", "J", "L", "O", "S", "T", "Z") else "N"
                lines[(fi, game_id, player, t)] = (f"{fi}:{game_id}:{player}:{t} {bytes(r[3]).hex()} {piece(r[4])} "
                                                   f"{piece(r[5])} " + " ".join(piece(q) for q in r[6:11]) +
                                                   f" {int(r[11])} {int(r[12])} {int(r[13])}")
    return m.Games(rows_by_game), lines


def features(games, lines, tool, cache_dir, paths):
    key = hashlib.sha1()
    for p in paths + [tool]:
        st = os.stat(p)
        key.update(f"{os.path.abspath(p)}:{st.st_size}:{st.st_mtime_ns}".encode())
    cache = os.path.join(cache_dir, f"features-{key.hexdigest()[:12]}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["F"], list(z["names"])
    G, T = games.mask.shape
    wanted = []
    for g, gid in enumerate(games.ids):
        for t in range(games.lengths[g]):
            wanted.append(lines[gid + (t,)])
    out = subprocess.run([tool], input="\n".join(wanted) + "\n", capture_output=True, text=True, check=True).stdout
    rows = out.splitlines()
    names = rows[0].split()[1:]
    F = np.zeros((G, T, len(names)), dtype=np.float32)
    index = {gid: g for g, gid in enumerate(games.ids)}
    for row in rows[1:]:
        parts = row.split()
        fi, game_id, player, t = (int(x) for x in parts[0].split(":"))
        F[index[(fi, game_id, player)], t] = [float(x) for x in parts[1:]]
    np.savez(cache, F=F, names=np.array(names, dtype=object))
    return F, names


def expand(F, names, onehot=False):
    """Raw features plus 'attacks at all' indicators for the attack features, and with `onehot` the attack
    features as one-hot sizes 1..7+ (a linear encoder can't otherwise map 'the rollout sends 5' to a size)."""
    if onehot:
        X, xnames = expand(F, names)
        parts, part_names = [X], list(xnames)
        for i, n in enumerate(names):
            if "attack" in n:
                v = np.minimum(np.round(F[..., i]), m.ATTACKS - 1).astype(int)
                parts.append(np.eye(m.ATTACKS, dtype=np.float32)[v][..., 1:])
                part_names += [f"{n}={k}" for k in range(1, m.ATTACKS)]
        return np.concatenate(parts, -1), part_names
    extra, extra_names = [], []
    for i, n in enumerate(names):
        if "attack" in n:
            extra.append((F[..., i] > 0).astype(np.float32))
            extra_names.append(n + ">0")
    X = np.concatenate([F, np.stack(extra, axis=-1)], axis=-1) if extra else F
    return X, list(names) + extra_names


class HMM(torch.nn.Module):
    """The factored MarkovOpponent transition (W + U[action] + V[jeopardy] + X[since]) and free emissions, or a
    fixed full transition table."""

    def __init__(self, initial, emission, W=None, U=None, V=None, X=None, full=None, trainable=True):
        super().__init__()
        t = lambda a: torch.nn.Parameter(torch.tensor(np.asarray(a), dtype=torch.float64), requires_grad=trainable)
        self.init_logit = t(np.log(np.maximum(initial, 1e-9)))
        self.em_logit = t(np.log(np.maximum(emission, 1e-12)))
        if full is not None:
            self.register_buffer("full", torch.tensor(full, dtype=torch.float64))
        else:
            self.full = None
            self.W, self.U, self.V, self.X = t(W), t(U), t(V), t(X)

    def initial(self):
        return torch.softmax(self.init_logit, -1)

    def emission(self):
        return torch.softmax(self.em_logit, -1)

    def transition(self):
        if self.full is not None:
            return self.full
        logits = (self.W[None, None, None] + self.U[:, None, None, None, :] + self.V[None, :, None, None, :]
                  + self.X[None, None, :, None, :])
        return torch.softmax(logits, -1)


class Data:
    def __init__(self, games, jb, X, rows):
        g = games.subset(rows)
        self.G, self.T = g.mask.shape
        tt = lambda a, dt=torch.long: torch.tensor(np.asarray(a), dtype=dt)
        self.mask = tt(g.mask, torch.bool)
        self.fresh = tt(g.fresh, torch.bool)
        self.em = tt(g.emission)
        self.action = tt(g.action)
        self.jb = tt(jb[rows])
        self.since = tt(g.since)
        self.lengths = tt(g.lengths)
        self.X = tt(X[rows], torch.float64)
        self.steps = int(g.mask.sum())
        self.attack = g.attack


def filter_prior(hmm, d):
    """Prior over states at each step from the damage filter on the target's past attacks."""
    A = hmm.transition()
    B = hmm.emission()
    init = hmm.initial()
    priors = []
    prev = None
    for t in range(d.T):
        if t == 0:
            p = init.expand(d.G, S)
        else:
            At = A[d.action[:, t], d.jb[:, t], d.since[:, t]]
            p = torch.einsum("gi,gij->gj", prev, At)
            p = torch.where(d.fresh[:, t, None], init.expand(d.G, S), p)
        priors.append(p)
        post = p * B[:, d.em[:, t]].T
        post = post / post.sum(-1, keepdim=True).clamp_min(1e-300)
        prev = torch.where(d.mask[:, t, None], post, p)
    return torch.stack(priors, 1)


def root_emission(hmm, size):
    """Emissions with each root's attack-size shift: the chance of attacking at all and of dying stay the state's,
    the size of an attack is softmax(state's size logits + the root's shift). Shape (G, T, S, E)."""
    B = hmm.emission()
    if size is None:
        return None
    K = m.ATTACKS
    attack = B[:, 1:K].sum(-1)
    sizes = torch.softmax(hmm.em_logit[:, 1:K][None, None] + size[:, :, None, :], -1)
    G, T = size.shape[:2]
    return torch.cat([B[:, :1].expand(G, T, S, 1), attack[None, None, :, None] * sizes,
                      B[:, K:].expand(G, T, S, E - K)], -1)


def likelihood(B, Broot, e):
    """P(emission e | state) for every root: (G, T, S)."""
    if Broot is None:
        return B[:, e].permute(1, 2, 0)
    return torch.gather(Broot, 3, e[:, :, None, None].expand(*e.shape, S, 1)).squeeze(-1)


def horizon_loglik(hmm, d, root, horizon, size=None):
    """Log-likelihood of each root's next `horizon` attacks (summed), and the number of pieces it covers."""
    A = hmm.transition()
    B = hmm.emission()
    Broot = root_emission(hmm, size)
    init = hmm.initial()
    G, T = d.G, d.T
    b = root
    total = torch.zeros(G, T, dtype=torch.float64)
    count = torch.zeros(G, T, dtype=torch.float64)
    t_index = torch.arange(T)
    per_step = []
    for i in range(horizon):
        s = t_index + i
        valid = d.mask & (s[None, :] < d.lengths[:, None])
        sc = s.clamp(max=T - 1)
        e = d.em[:, sc]
        lik = likelihood(B, Broot, e)
        p = (b * lik).sum(-1)
        ll = torch.where(valid, torch.log(p.clamp_min(1e-300)), torch.zeros_like(p))
        per_step.append(ll)
        total = total + ll
        count = count + valid
        b = b * lik / p.clamp_min(1e-300)[..., None]
        if i + 1 < horizon:
            n = (sc + 1).clamp(max=T - 1)
            At = A[d.action[:, n], d.jb[:, n], d.since[:, n]]
            b = torch.einsum("gti,gtij->gtj", b, At)
            b = torch.where(d.fresh[:, n, None], init.expand(G, T, S), b)
    return total, count, per_step


def root_belief(hmm, d, enc):
    prior = filter_prior(hmm, d)
    if enc is None:
        return prior
    logit = torch.log(prior.clamp_min(1e-300)) + enc(d.X)
    return torch.softmax(logit, -1)


def size_shift(enc, d):
    return enc.size_logits(d.X) if enc is not None and enc.size is not None else None


def score(hmm, d, enc, horizons):
    with torch.no_grad():
        root = root_belief(hmm, d, enc)
        size = size_shift(enc, d)
        out = {}
        for h in horizons:
            total, count, steps = horizon_loglik(hmm, d, root, h, size)
            out[h] = float(-(total[d.mask].sum() / count[d.mask].sum()) / LOG2)
        t_index = torch.arange(d.T)
        out["by_step"] = []
        for i, ll in enumerate(steps):
            valid = d.mask & ((t_index + i)[None, :] < d.lengths[:, None])
            out["by_step"].append(float(-ll[valid].mean() / LOG2))
        Broot = root_emission(hmm, size)
        pred = root @ hmm.emission() if Broot is None else torch.einsum("gts,gtse->gte", root, Broot)
        p = pred[d.mask]
        out["entropy1"] = float(-(p * torch.log(p.clamp_min(1e-300))).sum(-1).mean() / LOG2)
        return out


class Encoder(torch.nn.Module):
    """Linear, or one tanh hidden layer shared by the state head and the size head."""

    def __init__(self, dim, mu, sd, hidden=0, size=False):
        super().__init__()
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float64))
        self.register_buffer("sd", torch.tensor(sd, dtype=torch.float64))
        self.trunk = (torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.Tanh()) if hidden
                      else torch.nn.Identity()).double()
        width = hidden or dim
        self.net = torch.nn.Linear(width, S).double()
        self.size = torch.nn.Linear(width, m.ATTACKS - 1).double() if size else None
        for head in (self.net, self.size):
            if head is not None:
                torch.nn.init.zeros_(head.weight)
                torch.nn.init.zeros_(head.bias)

    def forward(self, X):
        return self.net(self.trunk((X - self.mu) / self.sd))

    def size_logits(self, X):
        return self.size(self.trunk((X - self.mu) / self.sd))


class Init:
    """Starting parameters in the factored form, from an EM fit or a countdown chain."""

    def __init__(self, initial, emission, W, U, V, X):
        self.initial, self.emission, self.W, self.U, self.V, self.X = initial, emission, W, U, V, X


def countdown_init(games):
    """State 0 attacks now, state k (1..5) attacks in k pieces, state 6 is quiet, state 7 is dying."""
    K = m.ATTACKS
    counts = np.bincount(games.emission[games.mask], minlength=E).astype(float) + 0.5
    sizes = counts[1:K] / counts[1:K].sum()
    em = np.zeros((S, E))
    em[0, 1:K] = 0.97 * sizes
    em[0, 0] = 0.03
    for z in range(1, 7):
        em[z, 0] = 0.97
        em[z, 1:K] = 0.02 * sizes
    em[1:7, K] = 0.01
    em[0, K] = 0.0
    em[7, K] = 0.5
    em[7, 0] = 0.4
    em[7, 1:K] = 0.1 * sizes
    em = (em + 1e-4) / (em + 1e-4).sum(-1, keepdims=True)
    T0 = np.full((S, S), 0.01)
    for k in range(1, 6):
        T0[k, k - 1] = 0.8
    T0[0, 6] = 0.6
    T0[0, 5] = 0.2
    T0[0, 0] = 0.2
    T0[6, 6] = 0.85
    T0[6, 5] = 0.1
    T0[7, 7] = 0.8
    T0 = T0 / T0.sum(-1, keepdims=True)
    zeros = lambda n: np.zeros((n, S))
    return Init(np.full(S, 1.0 / S), em, np.log(T0), zeros(A_), zeros(J), zeros(D))


def from_model(model, trainable=True):
    return HMM(model.initial, model.emission, model.W, model.U, model.V, model.X, trainable=trainable)


def sharpened_tables(trans, emission, a):
    """The search's current sharpening (MarkovOpponent.cpp): rows to the power a, death mass kept."""
    tr = trans ** a
    tr = tr / tr.sum(-1, keepdims=True)
    em = emission.copy()
    K = m.ATTACKS
    before = em[:, :K].sum(-1, keepdims=True)
    em[:, :K] = em[:, :K] ** a
    em[:, :K] *= before / np.maximum(em[:, :K].sum(-1, keepdims=True), 1e-300)
    return tr, em


def sharpened(model, a):
    tr = model.transition() ** a
    tr = tr / tr.sum(-1, keepdims=True)
    em = model.emission.copy()
    before = em[:, :m.ATTACKS].sum(-1, keepdims=True)
    em[:, :m.ATTACKS] = em[:, :m.ATTACKS] ** a
    em[:, :m.ATTACKS] *= before / np.maximum(em[:, :m.ATTACKS].sum(-1, keepdims=True), 1e-300)
    return tr, em


def tilt_row(p, values, a, target):
    """p^a * exp(lam * values), renormalised, with lam chosen so the mean of `values` is `target`."""
    base = np.where(p > 0, p ** a, 0.0)
    if base.sum() <= 0:
        return p
    def mean(lam):
        w = base * np.exp(lam * (values - values.max()))
        return (w * values).sum() / w.sum(), w
    lo, hi = -50.0, 50.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if mean(mid)[0] < target:
            lo = mid
        else:
            hi = mid
    w = mean(0.5 * (lo + hi))[1]
    return w / w.sum()


def sharpen_preserving(trans, emission, values, a, global_lam=0.0):
    """Transitions: rows to the power a. Emissions: each row's attack bins to the power a, tilted to keep that
    row's expected attack, then all rows tilted by `global_lam` (one APP correction); death mass kept."""
    tr = trans ** a
    tr = tr / tr.sum(-1, keepdims=True)
    em = emission.copy()
    K = m.ATTACKS
    for z in range(S):
        mass = em[z, :K].sum()
        if mass <= 0:
            continue
        p = em[z, :K] / mass
        q = tilt_row(p, values, a, (p * values).sum())
        if global_lam:
            q = q * np.exp(global_lam * (values - values.max()))
            q = q / q.sum()
        em[z, :K] = q * mass
    return tr, em


def open_loop_app(hmm, d, root, horizon, values):
    """Expected attack per piece over the next `horizon` pieces from each root, without seeing the attacks
    (the search's view), using the logged inputs; and the logged attack per piece over the same pieces."""
    with torch.no_grad():
        A = hmm.transition()
        B = hmm.emission()
        init = hmm.initial()
        v = torch.tensor(np.append(values, 0.0), dtype=torch.float64)
        G, T = d.G, d.T
        t_index = torch.arange(T)
        b = root
        model, logged, count = 0.0, 0.0, 0.0
        attack = torch.tensor(d.attack, dtype=torch.float64)
        for i in range(horizon):
            s = t_index + i
            valid = d.mask & (s[None, :] < d.lengths[:, None])
            sc = s.clamp(max=T - 1)
            model += float(((b @ B) @ v)[valid].sum())
            logged += float(attack[:, sc][valid].sum())
            count += float(valid.sum())
            n = (sc + 1).clamp(max=T - 1)
            At = A[d.action[:, n], d.jb[:, n], d.since[:, n]]
            b = torch.einsum("gti,gtij->gtj", b, At)
            b = torch.where(d.fresh[:, n, None], init.expand(G, T, S), b)
        return model / count, logged / count


def train(hmm, enc, train_d, val_d, horizon, epochs, lr, l2, log):
    params = [p for p in hmm.parameters() if p.requires_grad] + (list(enc.parameters()) if enc else [])
    opt = torch.optim.Adam(params, lr=lr)
    best, best_state, stale = np.inf, None, 0
    for epoch in range(epochs):
        opt.zero_grad()
        root = root_belief(hmm, train_d, enc)
        total, count, _ = horizon_loglik(hmm, train_d, root, horizon, size_shift(enc, train_d))
        loss = -(total[train_d.mask].sum() / count[train_d.mask].sum())
        if enc is not None:
            loss = loss + l2 * sum((w ** 2).sum() for n, w in enc.named_parameters() if "weight" in n)
        loss.backward()
        opt.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            val = score(hmm, val_d, enc, [horizon])[horizon]
            log(f"  epoch {epoch}: train {float(loss) / LOG2:.4f}, validation {val:.4f} bits/piece")
            if val < best - 1e-4:
                best, stale = val, 0
                best_state = ({k: v.clone() for k, v in hmm.state_dict().items()},
                              {k: v.clone() for k, v in enc.state_dict().items()} if enc else None)
            else:
                stale += 1
                if stale >= 5:
                    break
    hmm.load_state_dict(best_state[0])
    if enc is not None:
        enc.load_state_dict(best_state[1])
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dbs", nargs="+")
    parser.add_argument("--tool", required=True, help="nana_opp_features executable")
    parser.add_argument("--player", type=int, choices=[1, 2], help="target player in every database")
    parser.add_argument("--out", default=os.path.join("results", "encoder-hmm"))
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--test-share", type=float, default=0.2)
    parser.add_argument("--val-share", type=float, default=0.15)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--l2", type=float, default=1e-3, help="encoder weight decay")
    parser.add_argument("--hidden", type=int, default=0, help="encoder hidden units (0 = linear)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--states", type=int, default=m.STATES,
                        help="hidden states (MarkovOpponent.hpp's STATES must match to use the result in nana)")
    parser.add_argument("--init", choices=["em", "countdown"], default="em",
                        help="joint training starts from the EM refit or from a countdown chain")
    parser.add_argument("--onehot", action="store_true", help="also give the attack features as one-hot sizes")
    parser.add_argument("--size-head", action="store_true", help="the encoder also shifts the root's attack sizes")
    parser.add_argument("--no-sweep", action="store_true", help="skip the sharpening sweep")
    parser.add_argument("--load", metavar="DIR", help="use DIR/joint.pt instead of training (same data and seed)")
    args = parser.parse_args()
    global S
    S = m.STATES = args.states
    if args.init == "countdown" and S != 8:
        parser.error("the countdown start is laid out for 8 states")
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    log_file = open(os.path.join(args.out, "log.txt"), "w")

    def log(text):
        print(text, flush=True)
        log_file.write(text + "\n")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    games, lines = load(args.dbs, args.player)
    log(f"{len(games.ids)} sequences, {games.steps()} pieces, {int(games.dead[games.mask].sum())} deaths; "
        f"attack/piece {games.attack[games.mask].mean():.3f}")
    F, names = features(games, lines, args.tool, args.out, args.dbs)
    X, xnames = expand(F, names, args.onehot)

    meter = np.where(np.isnan(games.meter_before), 0, games.meter_before)
    _, limits = m.choose_jeopardy(games, meter)
    jb = m.bucket(meter, limits)
    log(f"jeopardy limits {limits}")

    order = rng.permutation(len(games.ids))
    n_test = int(round(args.test_share * len(order)))
    n_val = int(round(args.val_share * len(order)))
    test_rows, val_rows, train_rows = (np.sort(order[:n_test]), np.sort(order[n_test:n_test + n_val]),
                                       np.sort(order[n_test + n_val:]))
    log(f"games: {len(train_rows)} train, {len(val_rows)} validation, {len(test_rows)} test")
    train_d, val_d, test_d = (Data(games, jb, X, r) for r in (train_rows, val_rows, test_rows))
    horizons = sorted({1, 5, args.horizon})
    results = {}

    def report(name, hmm, enc):
        r = score(hmm, test_d, enc, horizons)
        results[name] = r
        log(f"{name:12s} " + "  ".join(f"h{h} {r[h]:.4f}" for h in horizons) + f"  entropy(next) {r['entropy1']:.4f}")
        log(f"{'':12s} bits at step " + " ".join(f"{x:.3f}" for x in r["by_step"]))

    train_games = games.subset(train_rows)
    counts = np.bincount(train_games.emission[train_games.mask], minlength=E) + 0.5
    pooled = HMM(np.full(S, 1.0 / S), np.tile(counts / counts.sum(), (S, 1)), full=np.broadcast_to(
        np.eye(S), (A_, J, D, S, S)).copy(), trainable=False)

    def report_on(name, hmm, d):
        r = score(hmm, d, None, horizons)
        results[name] = r
        log(f"{name:12s} " + "  ".join(f"h{h} {r[h]:.4f}" for h in horizons) + f"  entropy(next) {r['entropy1']:.4f}")
        log(f"{'':12s} bits at step " + " ".join(f"{x:.3f}" for x in r["by_step"]))

    log(f"\nheld out ({len(test_rows)} games, {test_d.steps} pieces), bits per piece over the next h attacks:")
    report("pooled", pooled, None)
    if S == 8:
        shipped = m.LoadedModel(SHIPPED)
        ship_test = Data(games, m.bucket(meter, shipped.limits), X, test_rows)
        report_on("shipped", HMM(shipped.initial, shipped.emission, full=shipped.transition(), trainable=False),
                  ship_test)
        tr4, em4 = sharpened(shipped, 4.0)
        report_on("shipped a=4", HMM(shipped.initial, em4, full=tr4, trainable=False), ship_test)

    log("\nrefitting the HMM by EM on the training games")
    model = m.fit(train_games, jb[train_rows], rng, restarts=args.restarts, log=lambda s: None)
    refit = from_model(model, trainable=False)
    report("refit", refit, None)

    *_, gamma = m.forward_backward(model, train_games, jb[train_rows], return_gamma=True)
    Xtr = X[train_rows][train_games.mask]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    two = Encoder(X.shape[-1], mu, sd)
    import board_encoder
    W, b, _ = board_encoder.fit_encoder((Xtr - mu) / sd, gamma[train_games.mask])
    marginal = gamma[train_games.mask].mean(0)
    with torch.no_grad():
        two.net.weight.copy_(torch.tensor(W.T))
        two.net.bias.copy_(torch.tensor(b - np.log(marginal + 1e-12)))
    report("two-stage", refit, two)

    log(f"\ntraining HMM and encoder together on the {args.horizon}-attack loss")
    start = model if args.init == "em" else countdown_init(train_games)
    log(f"  starting from the {args.init} parameters" + (", with a per-root attack-size head" if args.size_head else ""))
    joint = from_model(start, trainable=True)
    enc = Encoder(X.shape[-1], mu, sd, args.hidden, args.size_head)
    if args.load:
        saved = torch.load(os.path.join(args.load, "joint.pt"), weights_only=False)
        joint.load_state_dict(saved["hmm"])
        enc.load_state_dict(saved["encoder"])
        log(f"  loaded {args.load} instead of training")
    else:
        train(joint, enc, train_d, val_d, args.horizon, args.epochs, args.lr, args.l2, log)
    report("joint", joint, enc)

    log("\njoint encoder weights (standardised feature x state):")
    if not args.hidden:
        Wj = enc.net.weight.detach().numpy().T
        for i, n in enumerate(xnames):
            log(f"  {n:22s} " + " ".join(f"{w:+.2f}" for w in Wj[i]))
    em = joint.emission().detach().numpy()
    log("joint emissions (state x attack bin 0..7, death):")
    for z in range(S):
        log(f"  z{z} " + " ".join(f"{p:.3f}" for p in em[z]))

    if args.no_sweep:
        torch.save({"hmm": joint.state_dict(), "encoder": enc.state_dict(), "limits": limits, "features": xnames,
                    "hidden": args.hidden, "args": vars(args)}, os.path.join(args.out, "joint.pt"))
        with open(os.path.join(args.out, "report.json"), "w") as f:
            json.dump({"results": results, "limits": list(limits), "args": vars(args)}, f, indent=1)
        log(f"wrote {args.out}")
        return
    values = np.arange(m.ATTACKS, dtype=float)
    alive = train_games.mask & ~train_games.dead
    for e in range(m.ATTACKS):
        sel = alive & (train_games.emission == e)
        if sel.any():
            values[e] = train_games.attack[sel].mean()
    log("\nsharpening the joint model (attack values " + ", ".join(f"{v:.2f}" for v in values) + ")")
    with torch.no_grad():
        base_tr = joint.transition().numpy()
        base_em = joint.emission().numpy()
        base_init = joint.initial().numpy()
        train_root = root_belief(joint, train_d, enc)
    data_app = open_loop_app(joint, train_d, train_root, args.horizon, values)[1]
    log(f"{'a':>4s} {'method':10s} " + " ".join(f"{'h' + str(h):>7s}" for h in horizons) +
        f" {'entropy':>8s} {'APP':>6s} (data {data_app:.3f}, test data "
        f"{open_loop_app(joint, test_d, root_belief(joint, test_d, enc), args.horizon, values)[1]:.3f})")
    sweep = {}
    for a in (1.0, 1.5, 2.0, 3.0, 4.0):
        methods = {"power": sharpened_tables(base_tr, base_em, a)}
        tr, em = sharpen_preserving(base_tr, base_em, values, a)
        lo, hi = -5.0, 5.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            tr_m, em_m = sharpen_preserving(base_tr, base_em, values, a, mid)
            h = HMM(base_init, em_m, full=tr_m, trainable=False)
            app = open_loop_app(h, train_d, root_belief(h, train_d, enc), args.horizon, values)[0]
            if app < data_app:
                lo = mid
            else:
                hi = mid
        methods["keep-APP"] = sharpen_preserving(base_tr, base_em, values, a, 0.5 * (lo + hi))
        for name, (tr_x, em_x) in methods.items():
            h = HMM(base_init, em_x, full=tr_x, trainable=False)
            r = score(h, test_d, enc, horizons)
            app = open_loop_app(h, test_d, root_belief(h, test_d, enc), args.horizon, values)[0]
            sweep[f"{a}-{name}"] = dict(r, app=app)
            log(f"{a:4.1f} {name:10s} " + " ".join(f"{r[hh]:7.4f}" for hh in horizons) +
                f" {r['entropy1']:8.4f} {app:6.3f}")
    results["sharpening"] = sweep

    torch.save({"hmm": joint.state_dict(), "encoder": enc.state_dict(), "limits": limits, "features": xnames,
                "hidden": args.hidden, "args": vars(args)}, os.path.join(args.out, "joint.pt"))
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump({"results": results, "limits": list(limits), "args": vars(args),
                   "games": {"train": len(train_rows), "validation": len(val_rows), "test": len(test_rows)}}, f,
                  indent=1)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
