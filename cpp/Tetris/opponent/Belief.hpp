#pragma once
#include "IOHMM.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace opponent {
class Belief {
public:
    Belief() { reset(); }

    void reset() {
        fresh = true;
        joint = {};
        our_meter = 0;
        our_last_attack = 0;
    }

    bool started() const { return !fresh; }

    static Belief from_root(const Root& root) {
        Belief f;
        f.fresh = root.fresh;
        for (int z = 0; z < opponent::STATES; z++) {
            for (int s = 0; s < opponent::SINCE; s++) {
                f.joint[z][s] = root.state[z] * root.since[s];
            }
        }
        f.normalise();
        return f;
    }

    void apply_evidence(const std::array<float, opponent::STATES>* state_evidence,
                       const std::array<float, opponent::SINCE>* since_evidence) {
        if (fresh) {
            return;
        }
        const auto before = joint;
        for (int z = 0; z < opponent::STATES; z++) {
            for (int s = 0; s < opponent::SINCE; s++) {
                if (state_evidence) joint[z][s] *= (*state_evidence)[z];
                if (since_evidence) joint[z][s] *= (*since_evidence)[s];
            }
        }
        if (!normalise()) {
            joint = before;
        }
    }

    void update(int our_attack, bool we_cleared, int received) {
        const opponent::IOHMMParams& p = opponent::iohmm_params();
        const int action = std::min(our_last_attack, opponent::ACTIONS - 1);

        std::array<std::array<float, opponent::SINCE>, opponent::STATES> predicted{};
        if (fresh) {
            for (int z = 0; z < opponent::STATES; z++) {
                predicted[z][0] = opponent::probability(p.initial, z);
            }
        }
        else {
            for (int z = 0; z < opponent::STATES; z++) {
                for (int s = 0; s < opponent::SINCE; s++) {
                    const float mass = joint[z][s];
                    if (mass <= 0) {
                        continue;
                    }
                    const auto& row = p.transition[action][0][s][z];
                    for (int to = 0; to < opponent::STATES; to++) {
                        predicted[to][s] += mass * opponent::probability(row, to);
                    }
                }
            }
        }

        std::array<float, opponent::EMISSIONS> like{};
        for (int e = 0; e < opponent::ATTACKS; e++) {
            like[e] = we_cleared ? 1.0f : landing(e, our_attack) == received ? 1.0f : 0.0f;
        }
        if (!we_cleared && received > 0) {
            const int at_top = landing(opponent::ATTACKS - 1, our_attack);
            if (received >= at_top) {
                like[opponent::ATTACKS - 1] = 1.0f;
            }
        }

        std::array<std::array<float, opponent::SINCE>, opponent::STATES> next{};
        std::array<float, opponent::EMISSIONS> emitted{};
        float total = 0;
        for (int z = 0; z < opponent::STATES; z++) {
            for (int s = 0; s < opponent::SINCE; s++) {
                const float mass = predicted[z][s];
                if (mass <= 0) {
                    continue;
                }
                for (int e = 0; e < opponent::ATTACKS; e++) {
                    const float w = mass * opponent::probability(p.emission[z], e) * like[e];
                    if (w <= 0) {
                        continue;
                    }
                    const int s2 = e == 0 ? std::min(s + 1, opponent::SINCE - 1) : 0;
                    next[z][s2] += w;
                    emitted[e] += w;
                    total += w;
                }
            }
        }
        if (total <= 0) {
            if (!we_cleared) {
                update(our_attack, true, 0);
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
            for (int e = 0; e < opponent::ATTACKS; e++) {
                expected += emitted[e] / total * std::max(0.0f, our_meter + p.attack_value[e] - our_attack);
            }
            our_meter = expected;
        }
        our_last_attack = our_attack;
        fresh = false;
    }

    Root root(const std::array<float, opponent::STATES>* state_evidence = nullptr,
                      const std::array<float, opponent::SINCE>* since_evidence = nullptr) const {
        Root out;
        out.fresh = fresh;
        if (fresh) {
            return out;
        }
        std::array<std::array<float, opponent::SINCE>, opponent::STATES> j = joint;
        float total = 0;
        for (int z = 0; z < opponent::STATES; z++) {
            for (int s = 0; s < opponent::SINCE; s++) {
                if (state_evidence) j[z][s] *= (*state_evidence)[z];
                if (since_evidence) j[z][s] *= (*since_evidence)[s];
                total += j[z][s];
            }
        }
        if (total <= 0) {
            j = joint;
            total = 1;
        }
        for (int z = 0; z < opponent::STATES; z++) {
            for (int s = 0; s < opponent::SINCE; s++) {
                out.state[z] += j[z][s] / total;
                out.since[s] += j[z][s] / total;
            }
        }
        return out;
    }

    std::array<float, opponent::STATES> predicted() const {
        std::array<float, opponent::STATES> out{};
        const opponent::IOHMMParams& p = opponent::iohmm_params();
        if (fresh) {
            for (int z = 0; z < opponent::STATES; z++) {
                out[z] = opponent::probability(p.initial, z);
            }
            return out;
        }
        const int action = std::min(our_last_attack, opponent::ACTIONS - 1);
        for (int z = 0; z < opponent::STATES; z++) {
            for (int s = 0; s < opponent::SINCE; s++) {
                const float mass = joint[z][s];
                if (mass <= 0) {
                    continue;
                }
                const auto& row = p.transition[action][0][s][z];
                for (int to = 0; to < opponent::STATES; to++) {
                    out[to] += mass * opponent::probability(row, to);
                }
            }
        }
        return out;
    }

    template <class Weights>
    float value_after(int our_attack, bool we_cleared, int received, const Weights& W) const {
        Belief f = *this;
        f.update(our_attack, we_cleared, received);
        const std::array<float, opponent::STATES> q = f.predicted();
        float value = 0;
        for (int z = 0; z < opponent::STATES; z++) {
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

    std::array<std::array<float, opponent::SINCE>, opponent::STATES> joint{};
    bool fresh = true;
    float our_meter = 0;
    int our_last_attack = 0;

    int landing(int e, int our_attack) const {
        const float o = opponent::iohmm_params().attack_value[e];
        return (int)std::lround(std::max(0.0f, our_meter + o - (float)our_attack));
    }
};
}
