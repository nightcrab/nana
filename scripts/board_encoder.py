"""Does the opponent's board tell us its HMM state beyond what its damage does?

Two-stage experiment on stadium games, before committing to anything bigger:
  1. the fitted MarkovOpponent (damage only, unchanged) gives smoothed posteriors gamma_t(z) over each
     player's hidden state, from forward-backward over the whole game;
  2. an encoder q(z | board_t), softmax regression on a few board features, is trained on those soft
     labels (cross-entropy) on 80% of the games;
  3. on the other 20%, beliefs over z_t at each piece are compared by how well they predict what
     happens next:
       prior      the states' average posterior: knows nothing
       filter     the damage-only forward filter, P(z_t | attacks before t): what a root filter knows
       encoder    q(z_t | board_t): the board alone
       combined   filter * encoder / prior: both (board and history assumed independent given z)
     scored by the log-likelihood of the next 1 and 5 emissions (attack bins or death), and of whether
     the player dies within 10 pieces (the quantity the search's kill reward uses).

It also fits a second head, q(since_t | board_t): the opponent's pieces-since-its-last-attack counter,
an input to the HMM that the search's root can't observe (logged directly, so hard labels).

usage: python scripts/board_encoder.py STADIUM.db [...] [--model cpp/Tetris/game/MarkovOpponentParams.hpp]
                                       [--out cpp/Tetris/game/OpponentEncoderParams.hpp]
       python scripts/board_encoder.py --placeholder [--out ...]     an encoder nana switches off
--out writes the encoder for nana (cpp/Tetris/game/OpponentEncoder.hpp), fitted on all the games after
the held-out report. --parity FILE writes sample boards with their features, for checking that nana
computes the same features (cpp OpponentEncoder::boardFeatures).
"""

import argparse
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import train_markov_opponent as m

# the order nana's OpponentEncoder::boardFeatures computes them in
FEATURES = ["max_height", "mean_height", "min_height", "holes", "covered", "bumpiness", "max_well",
            "row_transitions", "combo", "b2b", "top_quarter"]


def board_features(blob, combo, b2b):
    b = np.frombuffer(blob, dtype=np.uint8).reshape(20, 10).astype(bool)  # [y][x], y = 0 at the bottom
    filled_rows = b.any(axis=1)
    heights = np.array([(np.max(np.nonzero(b[:, x])[0]) + 1) if b[:, x].any() else 0 for x in range(10)])
    holes = 0
    covered = 0
    for x in range(10):
        h = heights[x]
        empty_below = h - int(b[:h, x].sum())
        holes += empty_below
        # filled cells above the column's lowest hole
        col = b[:h, x]
        if empty_below:
            lowest_hole = int(np.argmin(col))
            covered += int(col[lowest_hole:].sum())
    bumpiness = int(np.abs(np.diff(heights)).sum())
    padded = np.concatenate([[99], heights, [99]])
    wells = [min(padded[i - 1], padded[i + 1]) - padded[i] for i in range(1, 11)]
    row_transitions = int((b[:, 1:] != b[:, :-1]).sum())
    lowest = int(heights.min())
    return [heights.max(), heights.mean(), lowest, holes, covered, bumpiness, max(0, max(wells)),
            row_transitions, combo, b2b, int(heights.max() >= 15)]


def load_boards(paths, players=(1, 2), raw=None):
    """{(file_index, game_id, player): [features per step]}, steps exactly as load_stadium builds them.
    raw: a list to collect (board blob, combo, b2b, features) samples into, for --parity."""
    out = {}
    for file_index, path in enumerate(paths):
        db = sqlite3.connect(path)
        rows_by_id = defaultdict(list)
        for row in db.execute("SELECT game_id, move_index, state, p1_board, p1_combo, p1_b2b, p2_board, p2_combo, "
                              "p2_b2b FROM Data ORDER BY game_id, move_index"):
            rows_by_id[row[0]].append(row)
        for game_id, rows in rows_by_id.items():
            playing = [r for r in rows if r[2] == "PLAYING"]
            final = [r for r in rows if r[2] != "PLAYING"]
            if not playing or len(final) != 1:
                continue
            result = final[0][2]
            for player in players:
                cols = (3, 4, 5) if player == 1 else (6, 7, 8)
                lost = result == "DRAW" or result == ("P2_WIN" if player == 1 else "P1_WIN")
                feats = [board_features(r[cols[0]], r[cols[1]], r[cols[2]]) for r in playing]
                if raw is not None and len(raw) < 5000:
                    raw += [(r[cols[0]], r[cols[1]], r[cols[2]], f) for r, f in zip(playing, feats)]
                if lost:
                    f = final[0]
                    feats.append(board_features(f[cols[0]], f[cols[1]], f[cols[2]]))
                out[(file_index, game_id, player)] = feats
    return out


def fingerprint(model):
    """sum over the emission table (rows renormalised) of (index + 1) * p, as nana's hmmFingerprint"""
    flat = model.emission.reshape(-1)
    return float(np.sum(np.arange(1, len(flat) + 1) * flat))


def fmt(x):
    return m.fmt(float(x))


def write_header(path, fp, mean, scale, w_state, b_state, w_since, b_since, state_prior, since_prior, source):
    def arr(values):
        return m.braces(fmt(v) for v in values)

    def mat(rows):
        return "{{\n" + ",\n".join("        " + arr(r) for r in rows) + "\n    }}"

    lines = [f"// generated by scripts/board_encoder.py from {source}; don't edit",
             "#pragma once",
             '#include "OpponentEncoder.hpp"',
             "",
             "namespace markov {",
             f"static_assert(STATES == {m.STATES} && SINCE == {m.SINCE} && ENCODER_FEATURES == {len(FEATURES)}, "
             '"OpponentEncoderParams.hpp was fitted with other dimensions");',
             "",
             "inline constexpr EncoderParams fitted_encoder = {",
             f"    // fingerprint of the HMM it was trained against (0 = no encoder)",
             f"    {fp!r},",
             "    // feature mean, scale: " + ", ".join(FEATURES),
             "    " + arr(mean) + ",",
             "    " + arr(scale) + ",",
             "    // state head: weights [feature][state], bias",
             "    " + mat(w_state) + ",",
             "    " + arr(b_state) + ",",
             "    // since head: weights [feature][since], bias",
             "    " + mat(w_since) + ",",
             "    " + arr(b_since) + ",",
             "    // the heads' average outputs over the training data (state, since)",
             "    " + arr(state_prior) + ",",
             "    " + arr(since_prior) + ",",
             "};",
             "",
             "}"]
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def fit_encoder(X, gamma, l2=1e-3, steps=2000, lr=0.05):
    """Softmax regression on soft labels: minimise -sum gamma log softmax(X W + b), with Adam."""
    n, d = X.shape
    S = gamma.shape[1]
    W = np.zeros((d, S))
    b = np.log(gamma.mean(axis=0) + 1e-9)
    params = [W, b]
    mo = [np.zeros_like(p) for p in params]
    ve = [np.zeros_like(p) for p in params]
    for k in range(1, steps + 1):
        P = softmax(X @ W + b)
        G = (P - gamma) / n
        grads = [X.T @ G + l2 * W, G.sum(axis=0)]
        for i, (p, g) in enumerate(zip(params, grads)):
            mo[i] = 0.9 * mo[i] + 0.1 * g
            ve[i] = 0.999 * ve[i] + 0.001 * g * g
            p -= lr * (mo[i] / (1 - 0.9 ** k)) / (np.sqrt(ve[i] / (1 - 0.999 ** k)) + 1e-8)
    P = softmax(X @ W + b)
    loss = -np.mean(np.sum(gamma * np.log(P + 1e-12), axis=1))
    return W, b, loss


def filter_priors(model, games, jb):
    """P(z_t | emissions before t, inputs up to t) for every step: the damage-only filter's belief."""
    G, T = games.mask.shape
    A = model.transition()
    B = model.emission
    lik = B[:, games.emission].transpose(1, 2, 0)
    pri = np.zeros((G, T, m.STATES))
    prev = None
    for t in range(T):
        if t == 0:
            p = np.broadcast_to(model.initial, (G, m.STATES)).copy()
        else:
            At = A[games.action[:, t], jb[:, t], games.since[:, t]]
            p = np.einsum("gi,gij->gj", prev, At)
            p = np.where(games.fresh[:, t, None], model.initial[None], p)
        pri[:, t] = p
        post = p * lik[:, t]
        post /= np.maximum(post.sum(axis=1, keepdims=True), 1e-300)
        prev = post
    return pri


def received_filter(model, games):
    """The filter nana runs under TBP1 (cpp OpponentFilter), on stadium games: for each player (the
    opponent), from the other player's (our) view: our attack each turn, whether our piece cleared lines
    (approximated by our attack being positive: the databases don't log line clears), and the garbage
    we received under the stadium's netting. Returns P(state of each piece | what we saw before it),
    like filter_priors but without ever seeing the opponent's attack directly."""
    G, T = games.mask.shape
    S, D = m.STATES, m.SINCE
    A = model.transition()  # [action, jeopardy, since, from, to]
    B = model.emission       # [state, emission]
    value = model.attack_value
    # our attack on turn t is the opponent's input at its piece t + 1
    ours = np.zeros((G, T))
    ours[:, :-1] = games.sent[:, 1:]
    cleared = ours > 0
    o = games.attack
    # what we'd receive: track our true meter with the opponent's true attack
    received = np.zeros((G, T))
    meter = np.zeros(G)
    for t in range(T):
        land = np.maximum(0.0, meter + o[:, t] - ours[:, t])
        received[:, t] = np.where(cleared[:, t], 0.0, np.round(land))
        meter = np.where(cleared[:, t], land, 0.0)

    belief = np.zeros((G, T, S))
    joint = np.zeros((G, S, D))
    est_meter = np.zeros(G)
    last_ours = np.zeros(G, dtype=int)
    for t in range(T):
        fresh = games.fresh[:, t]
        # predict
        pred = np.zeros((G, S, D))
        for sc in range(D):
            At = A[np.minimum(last_ours, m.ACTIONS - 1), 0, sc]  # [G, from, to]
            pred[:, :, sc] = np.einsum("gi,gij->gj", joint[:, :, sc], At)
        init = np.zeros((S, D))
        init[:, 0] = model.initial
        pred = np.where(fresh[:, None, None], init[None], pred)
        belief[:, t] = pred.sum(axis=2)
        # likelihood of what we saw for each emission
        like = np.zeros((G, m.EMISSIONS))
        for e in range(m.ATTACKS):
            land = np.round(np.maximum(0.0, est_meter + value[e] - ours[:, t]))
            ok = land == received[:, t]
            if e == m.ATTACKS - 1:
                ok |= (received[:, t] > 0) & (received[:, t] >= land)
            like[:, e] = np.where(cleared[:, t], 1.0, ok.astype(float))
        # update over (state, emission), moving since on
        w = pred[:, :, :, None] * (B[None, :, None, :] * like[:, None, None, :])  # [G, S, D, E]
        nxt = np.zeros((G, S, D))
        nxt[:, :, 0] += w[:, :, :, 1:m.ATTACKS].sum(axis=(2, 3))
        stay = w[:, :, :, 0]
        nxt[:, :, 1:] += stay[:, :, :-1]
        nxt[:, :, D - 1] += stay[:, :, D - 1]
        total = nxt.sum(axis=(1, 2))
        ok = total > 0
        # when nothing fits, fall back to the prediction (learn nothing from the emission)
        nxt = np.where(ok[:, None, None], nxt / np.where(ok, total, 1)[:, None, None], pred)
        emitted = w.sum(axis=(1, 2)) / np.where(ok, total, 1)[:, None]
        expected_land = sum(emitted[:, e] * np.maximum(0.0, est_meter + value[e] - ours[:, t]) for e in range(m.ATTACKS))
        est_meter = np.where(cleared[:, t], expected_land, 0.0)
        joint = nxt
        last_ours = ours[:, t].astype(int)
    return belief


def k_step_loglik(model, games, jb, belief, k):
    """log P(emissions t..t+k-1 | belief over z_t, the inputs), for every step with k steps ahead in
    the same game (others NaN)."""
    G, T = games.mask.shape
    A = model.transition()
    B = model.emission
    lik = B[:, games.emission].transpose(1, 2, 0)
    out = np.full((G, T), np.nan)
    for t in range(T):
        ok = games.mask[:, t] & (t + k - 1 < games.lengths)
        if not ok.any():
            continue
        b = belief[ok, t]
        total = np.zeros(ok.sum())
        for i in range(k):
            ti = t + i
            p = b * lik[ok, ti]
            s = p.sum(axis=1)
            total += np.log(np.maximum(s, 1e-300))
            b = p / np.maximum(s[:, None], 1e-300)
            if i + 1 < k:
                nxt = ti + 1
                At = A[games.action[ok, nxt], jb[ok, nxt], games.since[ok, nxt]]
                b = np.einsum("gi,gij->gj", b, At)
                b = np.where(games.fresh[ok, nxt, None], model.initial[None], b)
        out[ok, t] = total
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dbs", nargs="*")
    parser.add_argument("--out", help="write the encoder header for nana")
    parser.add_argument("--placeholder", action="store_true", help="write an encoder nana switches off")
    parser.add_argument("--parity", help="write sample boards and their features to this file")
    parser.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "..", "cpp", "Tetris", "game",
                                                        "MarkovOpponentParams.hpp"))
    parser.add_argument("--test-share", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    out_path = args.out or os.path.join(os.path.dirname(__file__), "..", "cpp", "Tetris", "game",
                                        "OpponentEncoderParams.hpp")
    if args.placeholder:
        F_, S_, D_ = len(FEATURES), m.STATES, m.SINCE
        write_header(out_path, 0.0, [0.0] * F_, [1.0] * F_, [[0.0] * S_] * F_, [0.0] * S_, [[0.0] * D_] * F_,
                     [0.0] * D_, [1.0 / S_] * S_, [1.0 / D_] * D_, "nothing (placeholder: the encoder is off)")
        print(f"wrote {out_path}")
        return
    if not args.dbs:
        parser.error("no stadium databases")

    model = m.LoadedModel(args.model)
    games = m.load_any(args.dbs, (1, 2))
    jb = m.bucket(np.where(np.isnan(games.meter_before), 0, games.meter_before), model.limits)
    print(f"{len(games.ids)} sequences, {games.steps()} pieces; model {os.path.basename(args.model)}")

    raw = [] if args.parity else None
    boards = load_boards(args.dbs, raw=raw)
    if args.parity:
        with open(args.parity, "w") as f:
            for blob, combo, b2b, feats in raw:
                f.write(f"{bytes(blob).hex()} {combo} {b2b} " + " ".join(repr(float(x)) for x in feats) + "\n")
        print(f"wrote {len(raw)} samples to {args.parity}")
    G, T = games.mask.shape
    F = np.zeros((G, T, len(FEATURES)))
    for g, gid in enumerate(games.ids):
        feats = boards[gid]
        assert len(feats) == games.lengths[g], (gid, len(feats), games.lengths[g])
        F[g, :len(feats)] = feats

    *_, gamma = m.forward_backward(model, games, jb, return_gamma=True)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(G)
    n_test = int(round(args.test_share * G))
    test = np.zeros(G, dtype=bool)
    test[order[:n_test]] = True
    train_steps = games.mask & ~test[:, None]
    test_steps = games.mask & test[:, None]

    X_train = F[train_steps]
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0) + 1e-9
    norm = lambda X: (X - mu) / sd
    W, b, loss = fit_encoder(norm(X_train), gamma[train_steps])
    marginal = gamma[train_steps].mean(axis=0)
    base = -np.mean(np.sum(gamma[train_steps] * np.log(marginal + 1e-12), axis=1))
    print(f"encoder fit: cross-entropy {loss:.4f} nats against {base:.4f} for the states' average "
          f"(train); {1 - loss / base:.1%} of the uncertainty about z explained by the board")
    print("encoder weights (standardised features x states), largest per feature:")
    for i, name in enumerate(FEATURES):
        print(f"  {name:20s} " + " ".join(f"{w:+.2f}" for w in W[i]))

    # the since head: hard labels, the counter as logged before each piece
    since_onehot = np.eye(m.SINCE)[games.since]
    W_since, b_since, loss_since = fit_encoder(norm(X_train), since_onehot[train_steps])
    since_marginal = since_onehot[train_steps].mean(axis=0)
    test_since = softmax(norm(F[test_steps]) @ W_since + b_since)
    ce_since = -np.mean(np.log(test_since[np.arange(len(test_since)), games.since[test_steps]] + 1e-12))
    ce_since_base = -np.mean(np.log(since_marginal[games.since[test_steps]] + 1e-12))
    acc = np.mean(test_since.argmax(axis=1) == games.since[test_steps])
    print(f"since head, held out: cross-entropy {ce_since:.4f} nats against {ce_since_base:.4f} for the counter's "
          f"distribution ({1 - ce_since / ce_since_base:.1%} explained); top guess right {acc:.0%}")

    enc = softmax(norm(F) @ W + b)
    filt = filter_priors(model, games, jb)
    prior = np.broadcast_to(marginal, filt.shape)
    combined = filt * enc / marginal
    combined /= combined.sum(axis=2, keepdims=True)
    received = received_filter(model, games)
    received_enc = received * enc / marginal
    received_enc /= received_enc.sum(axis=2, keepdims=True)
    beliefs = {"prior": prior, "filter": filt, "encoder": enc, "combined": combined,
               "received (TBP1)": received, "received+encoder": received_enc}

    # death within 10 pieces, from the installed model's death table? the header holds a discounted
    # table, so refit P(dies within 10) per state on the training games
    table, _ = m.death_value(model, games.subset(np.where(~test)[0]), jb[~test], horizon=10)
    G_, T_ = games.mask.shape
    dead = games.dead & games.mask
    until = np.full((G_, T_), 10 ** 6)
    for t in range(T_ - 2, -1, -1):
        until[:, t] = np.where(dead[:, t + 1], 1, np.where(until[:, t + 1] < 10 ** 6, until[:, t + 1] + 1, 10 ** 6))
    left = games.lengths[:, None] - 1 - np.arange(T_)[None, :]
    label = until <= 10
    known = (label | (left >= 10)) & test_steps & ~games.dead & (left >= 1)
    next_j = np.zeros_like(jb)
    next_j[:, :-1] = jb[:, 1:]
    next_s = np.zeros_like(games.since)
    next_s[:, :-1] = games.since[:, 1:]

    print(f"\nheld out ({n_test} sequences, {int(test_steps.sum())} pieces): mean log-likelihood per piece")
    print(f"{'belief':18s} {'next 1':>9s} {'next 5':>9s} {'death<=10':>10s}")
    for name, bel in beliefs.items():
        row = []
        for k in (1, 5):
            ll = k_step_loglik(model, games, jb, bel, k)
            row.append(np.nanmean(ll[test_steps]) / k)
        # P(dies within 10) = sum over z_t of belief(z_t) * the table for z_t and the next piece's buckets
        pd =np.sum(bel * table[:, next_j, next_s].transpose(1, 2, 0), axis=2)
        pd = np.clip(pd, 1e-6, 1 - 1e-6)
        dl = np.where(label, np.log(pd), np.log(1 - pd))[known].mean()
        row.append(dl)
        print(f"{name:18s} {row[0]:9.4f} {row[1]:9.4f} {row[2]:10.4f}")

    if args.out:
        # the encoder for nana: both heads refitted on every game
        X_all = F[games.mask]
        mu, sd = X_all.mean(axis=0), X_all.std(axis=0) + 1e-9
        Xn = (X_all - mu) / sd
        W_all, b_all, _ = fit_encoder(Xn, gamma[games.mask])
        Ws_all, bs_all, _ = fit_encoder(Xn, since_onehot[games.mask])
        source = f"{len(args.dbs)} stadium databases, model {os.path.basename(args.model)}"
        # the heads' average outputs, for turning an output into evidence (divide by these)
        state_prior = softmax(Xn @ W_all + b_all).mean(axis=0)
        since_prior = softmax(Xn @ Ws_all + bs_all).mean(axis=0)
        write_header(out_path, fingerprint(model), mu, sd, W_all, b_all, Ws_all, bs_all, state_prior, since_prior,
                     source)
        print(f"wrote {out_path} (HMM fingerprint {fingerprint(model):.6f})")


if __name__ == "__main__":
    main()
