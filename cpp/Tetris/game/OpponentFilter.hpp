#pragma once
#include "MarkovOpponent.hpp"
#include "engine/Board.hpp"

#include <algorithm>
#include <array>
#include <cmath>

// the forward filter over a real opponent's hidden state and since counter, from what a TBP bot sees
// without the opponent's board: its own attack each turn, whether its piece cleared lines, and the
// garbage it then receives. so under TBP1 the search's root still knows roughly where the opponent is,
// and with a board (TBP2) the encoder's evidence multiplies in at the root (OpponentFilter::root).
//
// the HMM is fitted on stadium games, so this follows the stadium's rules: both players move at once,
// each attack goes into the other's meter, the meters net against each other, and a player whose piece
// cleared no lines tanks its whole meter. one turn t, our attack d and the opponent's attack o (its
// emission): if our piece cleared nothing we receive G = max(0, M + o - d), M our pending garbage, which
// pins o down when G > 0 and bounds it when G = 0; if our piece cleared lines nothing lands and o goes
// unobserved. the opponent's own pending garbage is taken as 0 (it is in 98% of stadium turns), so its
// jeopardy bucket is 0
// the garbage the client pushed under our board: k if `after` is `expected` raised k rows over k garbage
// rows, full but for one hole column (the stadium's add_garbage), 0 if they're the same, -1 if neither
// (a new game, or a client with other garbage)
inline int garbageRows(const Board& expected, const Board& after) {
    for (int k = 0; k <= 20; k++) {
        const uint32_t low = k ? (1u << k) - 1 : 0;
        int holes = 0;
        bool ok = true;
        for (int x = 0; x < Board::width && ok; x++) {
            const uint32_t raised = (uint32_t)((uint64_t)expected.board[x] << k);
            const uint32_t col = after.board[x];
            if ((col & ~low) != raised) {
                ok = false;
            }
            const uint32_t bits = col & low;
            if (k && bits == 0) {
                holes++;
            }
            else if (bits != low) {
                ok = false;
            }
        }
        if (ok && (k == 0 || holes == 1)) {
            return k;
        }
    }
    return -1;
}

class OpponentFilter {
public:
    OpponentFilter() { reset(); }

    // a new game: the opponent hasn't played yet
    void reset() {
        fresh = true;
        joint = {};
        our_meter = 0;
        our_last_attack = 0;
    }

    bool started() const { return !fresh; }

    // a filter with no history behind it, believing the root's marginals as if independent (a game
    // joined midway, known only from the opponent's board)
    static OpponentFilter fromRoot(const RootOpponent& root) {
        OpponentFilter f;
        f.fresh = root.fresh;
        for (int z = 0; z < markov::STATES; z++) {
            for (int s = 0; s < markov::SINCE; s++) {
                f.joint[z][s] = root.state[z] * root.since[s];
            }
        }
        f.normalise();
        return f;
    }

    // multiply in the encoder's evidence from the opponent's board, as root() does
    void applyEvidence(const std::array<float, markov::STATES>* state_evidence,
                       const std::array<float, markov::SINCE>* since_evidence) {
        if (fresh) {
            return;
        }
        const auto before = joint;
        for (int z = 0; z < markov::STATES; z++) {
            for (int s = 0; s < markov::SINCE; s++) {
                if (state_evidence) joint[z][s] *= (*state_evidence)[z];
                if (since_evidence) joint[z][s] *= (*since_evidence)[s];
            }
        }
        if (!normalise()) {
            joint = before;
        }
    }

    // one turn: our attack (before cancelling), whether our piece cleared lines, and the garbage we then
    // received (only meaningful if we cleared nothing; 0 if none arrived)
    void turn(int our_attack, bool we_cleared, int received) {
        const markov::Params& p = markov::params();
        // the opponent's input this piece is our attack of the turn before (the stadium's timing)
        const int action = std::min(our_last_attack, markov::ACTIONS - 1);

        // predict: P(state of this piece, since counter before it)
        std::array<std::array<float, markov::SINCE>, markov::STATES> predicted{};
        if (fresh) {
            for (int z = 0; z < markov::STATES; z++) {
                predicted[z][0] = markov::probability(p.initial, z);
            }
        }
        else {
            for (int z = 0; z < markov::STATES; z++) {
                for (int s = 0; s < markov::SINCE; s++) {
                    const float mass = joint[z][s];
                    if (mass <= 0) {
                        continue;
                    }
                    const auto& row = p.transition[action][0][s][z];
                    for (int to = 0; to < markov::STATES; to++) {
                        predicted[to][s] += mass * markov::probability(row, to);
                    }
                }
            }
        }

        // the observation's likelihood for each emission (death can't be it: the game goes on)
        std::array<float, markov::EMISSIONS> like{};
        for (int e = 0; e < markov::ATTACKS; e++) {
            like[e] = we_cleared ? 1.0f : landing(e, our_attack) == received ? 1.0f : 0.0f;
        }
        if (!we_cleared && received > 0) {
            // the 7+ bin covers attacks above its mean value too
            const int at_top = landing(markov::ATTACKS - 1, our_attack);
            if (received >= at_top) {
                like[markov::ATTACKS - 1] = 1.0f;
            }
        }

        // update, and move the since counter on with the emission
        std::array<std::array<float, markov::SINCE>, markov::STATES> next{};
        std::array<float, markov::EMISSIONS> emitted{};
        float total = 0;
        for (int z = 0; z < markov::STATES; z++) {
            for (int s = 0; s < markov::SINCE; s++) {
                const float mass = predicted[z][s];
                if (mass <= 0) {
                    continue;
                }
                for (int e = 0; e < markov::ATTACKS; e++) {
                    const float w = mass * markov::probability(p.emission[z], e) * like[e];
                    if (w <= 0) {
                        continue;
                    }
                    const int s2 = e == 0 ? std::min(s + 1, markov::SINCE - 1) : 0;
                    next[z][s2] += w;
                    emitted[e] += w;
                    total += w;
                }
            }
        }
        if (total <= 0) {
            // nothing the model allows fits what we saw (e.g. a garbage count off the table): learn
            // nothing from the emission rather than lose the belief
            if (!we_cleared) {
                turn(our_attack, true, 0);
                our_meter = 0;
            }
            return;
        }
        for (auto& row : next) {
            for (float& x : row) {
                x /= total;
            }
        }
        joint = next;

        // our pending garbage: tanked if we cleared nothing, else what's expected to have stayed
        if (!we_cleared) {
            our_meter = 0;
        }
        else {
            float expected = 0;
            for (int e = 0; e < markov::ATTACKS; e++) {
                expected += emitted[e] / total * std::max(0.0f, our_meter + p.attack_value[e] - our_attack);
            }
            our_meter = expected;
        }
        our_last_attack = our_attack;
        fresh = false;
    }

    // what the search's root needs: the filter's belief over the opponent's state and since counter,
    // times the encoder's evidence from its board when there is one (state_evidence and since_evidence
    // are the encoder's outputs divided by its training averages, so the board isn't counted as prior)
    RootOpponent root(const std::array<float, markov::STATES>* state_evidence = nullptr,
                      const std::array<float, markov::SINCE>* since_evidence = nullptr) const {
        RootOpponent out;
        out.fresh = fresh;
        if (fresh) {
            return out;
        }
        std::array<std::array<float, markov::SINCE>, markov::STATES> j = joint;
        float total = 0;
        for (int z = 0; z < markov::STATES; z++) {
            for (int s = 0; s < markov::SINCE; s++) {
                if (state_evidence) j[z][s] *= (*state_evidence)[z];
                if (since_evidence) j[z][s] *= (*since_evidence)[s];
                total += j[z][s];
            }
        }
        if (total <= 0) {
            j = joint;
            total = 1;
        }
        for (int z = 0; z < markov::STATES; z++) {
            for (int s = 0; s < markov::SINCE; s++) {
                out.state[z] += j[z][s] / total;
                out.since[s] += j[z][s] / total;
            }
        }
        return out;
    }

    // P(the opponent's next hidden state), predicted through its next transition, whose input is our
    // last turn's attack (as the next turn() will take it)
    std::array<float, markov::STATES> predicted() const {
        std::array<float, markov::STATES> out{};
        const markov::Params& p = markov::params();
        if (fresh) {
            for (int z = 0; z < markov::STATES; z++) {
                out[z] = markov::probability(p.initial, z);
            }
            return out;
        }
        const int action = std::min(our_last_attack, markov::ACTIONS - 1);
        for (int z = 0; z < markov::STATES; z++) {
            for (int s = 0; s < markov::SINCE; s++) {
                const float mass = joint[z][s];
                if (mass <= 0) {
                    continue;
                }
                const auto& row = p.transition[action][0][s][z];
                for (int to = 0; to < markov::STATES; to++) {
                    out[to] += mass * markov::probability(row, to);
                }
            }
        }
        return out;
    }

    // P . W one piece ahead: the belief after a turn of ours (as turn() takes it), predicted through
    // the opponent's next transition; W weighs its hidden states
    template <class Weights>
    float valueAfter(int our_attack, bool we_cleared, int received, const Weights& W) const {
        OpponentFilter f = *this;
        f.turn(our_attack, we_cleared, received);
        const std::array<float, markov::STATES> q = f.predicted();
        float value = 0;
        for (int z = 0; z < markov::STATES; z++) {
            value += q[z] * (float)W[z];
        }
        return value;
    }

private:
    bool normalise() {
        float total = 0;
        for (const auto& row : joint) {
            for (float x : row) {
                total += x;
            }
        }
        if (total <= 0) {
            return false;
        }
        for (auto& row : joint) {
            for (float& x : row) {
                x /= total;
            }
        }
        return true;
    }

    // P(state of its last piece, since counter after it)
    std::array<std::array<float, markov::SINCE>, markov::STATES> joint{};
    bool fresh = true;
    // garbage pending against us that we haven't tanked (we cleared lines), in expectation
    float our_meter = 0;
    int our_last_attack = 0;

    // the garbage we'd receive this turn if the opponent's emission were e and we cleared nothing
    int landing(int e, int our_attack) const {
        const float o = markov::params().attack_value[e];
        return (int)std::lround(std::max(0.0f, our_meter + o - (float)our_attack));
    }
};
