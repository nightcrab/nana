#pragma once
#include "MarkovOpponent.hpp"
#include "engine/Board.hpp"

#include <algorithm>
#include <array>
#include <cmath>

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

    void reset() {
        fresh = true;
        joint = {};
        our_meter = 0;
        our_last_attack = 0;
    }

    bool started() const { return !fresh; }

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

    void turn(int our_attack, bool we_cleared, int received) {
        const markov::Params& p = markov::params();
        const int action = std::min(our_last_attack, markov::ACTIONS - 1);

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

        std::array<float, markov::EMISSIONS> like{};
        for (int e = 0; e < markov::ATTACKS; e++) {
            like[e] = we_cleared ? 1.0f : landing(e, our_attack) == received ? 1.0f : 0.0f;
        }
        if (!we_cleared && received > 0) {
            const int at_top = landing(markov::ATTACKS - 1, our_attack);
            if (received >= at_top) {
                like[markov::ATTACKS - 1] = 1.0f;
            }
        }

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

    std::array<std::array<float, markov::SINCE>, markov::STATES> joint{};
    bool fresh = true;
    float our_meter = 0;
    int our_last_attack = 0;

    int landing(int e, int our_attack) const {
        const float o = markov::params().attack_value[e];
        return (int)std::lround(std::max(0.0f, our_meter + o - (float)our_attack));
    }
};
