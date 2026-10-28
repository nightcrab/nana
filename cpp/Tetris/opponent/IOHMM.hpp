#pragma once
#include "Opponent.hpp"

#include <algorithm>
#include <array>
#include <cstdint>

namespace opponent {
constexpr int STATES = 8;
constexpr int ACTIONS = 6;
constexpr int JEOPARDY = 3;
constexpr int SINCE = 10;
constexpr int ATTACKS = 8;
constexpr int DEATH = ATTACKS;
constexpr int EMISSIONS = ATTACKS + 1;

struct IOHMMParams {
    std::array<float, STATES> initial;
    std::array<std::array<std::array<std::array<std::array<float, STATES>, STATES>, SINCE>, JEOPARDY>, ACTIONS> transition;
    std::array<std::array<float, EMISSIONS>, STATES> emission;
    std::array<float, ATTACKS> attack_value;
    std::array<float, JEOPARDY - 1> jeopardy_limits;
    std::array<std::array<std::array<float, SINCE>, JEOPARDY>, STATES> death_value;
    int horizon;
    float discount;
    bool stadium_rules;
};

const IOHMMParams& iohmm_params();

inline int jeopardy_bucket(float meter) {
    const IOHMMParams& p = iohmm_params();
    int bucket = 0;
    while (bucket < JEOPARDY - 1 && meter >= p.jeopardy_limits[bucket]) {
        bucket++;
    }
    return bucket;
}

template <size_t N>
inline float probability(const std::array<float, N>& cdf, int i) {
    return cdf[i] - (i > 0 ? cdf[i - 1] : 0.0f);
}

struct alignas(64) RootPrior {
    std::array<float, STATES> cdf{};
    std::array<float, SINCE> since_cdf{};
    bool active = false;
};
inline RootPrior root_prior;

// The opponent at the search root: the IOHMM belief over its hidden state, and its garbage meter.
struct Root {
    std::array<float, STATES> state{};
    std::array<float, SINCE> since{};
    bool fresh = true;
    float meter = 0;
    double last_attack = 0;

    void publish() const {
        root_prior.active = !fresh;
        toCdf(state, root_prior.cdf);
        toCdf(since, root_prior.since_cdf);
    }

private:
    template <size_t N>
    static void toCdf(const std::array<float, N>& p, std::array<float, N>& cdf) {
        float total = 0;
        for (float x : p) {
            total += x;
        }
        float sum = 0;
        for (size_t i = 0; i < N; i++) {
            sum += total > 0 ? p[i] / total : (i == 0 ? 1.0f : 0.0f);
            cdf[i] = sum;
        }
        cdf[N - 1] = 1.0f;
    }
};

class IOHMM {
public:
    static constexpr bool simulated = true;

    Prediction predict(float received, float meter, Rng& rng) {
        const IOHMMParams& p = iohmm_params();
        const int action = std::min<int>((int)received, ACTIONS - 1);
        const auto& row = fresh ? p.initial : p.transition[action][jeopardy_bucket(meter)][since][state];
        fresh = false;
        state = (uint8_t)rng.sample(row);
        const int emission = rng.sample(p.emission[state]);
        since = emission == 0 ? (uint8_t)std::min(since + 1, SINCE - 1) : 0;
        if (emission == DEATH) {
            return { 0, true };
        }
        return { p.attack_value[emission], false };
    }

    // A dead opponent restarts from the initial distribution.
    bool respawn() {
        fresh = true;
        return true;
    }

    void adopt(const Root& root) {
        state = 0;
        since = 0;
        fresh = root.fresh;
    }

    void sample_root(Rng& rng) {
        if (root_prior.active && !fresh) {
            state = (uint8_t)rng.sample(root_prior.cdf);
            since = (uint8_t)rng.sample(root_prior.since_cdf);
        }
    }

private:
    uint8_t state = 0;
    uint8_t since = 0;
    bool fresh = true;
};
}
