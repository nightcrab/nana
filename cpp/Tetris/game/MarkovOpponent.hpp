#pragma once
#include "Util/FastSeed.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#ifdef NANA_NEURAL_MODEL_DIR
#include "NeuralForecast.hpp"
#endif

namespace markov {
constexpr int STATES = 8;
constexpr int ACTIONS = 6;
constexpr int JEOPARDY = 3;
constexpr int SINCE = 10;
constexpr int ATTACKS = 8;
constexpr int DEATH = ATTACKS;
constexpr int EMISSIONS = ATTACKS + 1;
constexpr float GARBAGE_CAP = 8;

struct Params {
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

const Params& params();

inline int jeopardy_bucket(float meter) {
    const Params& p = params();
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

struct alignas(64) RootBelief {
    std::array<float, STATES> cdf{};
    std::array<float, SINCE> since_cdf{};
    bool active = false;
};
inline RootBelief root_belief;
}

struct RootOpponent;

class MarkovOpponent {
public:
    static constexpr bool simulated = true;

    MarkovOpponent() = default;

    float meter = 0;

    bool dead = false;
    double deaths = 0;
    float pieces = 0;
    float total_damage = 0;
    double last_attack = 0;

    // Advances internal state and returns outgoing damage this turn
    double play() {
        const markov::Params& p = markov::params();

#ifdef NANA_NEURAL_MODEL_DIR
        if (neural_terminal) return 0;
#endif

        if (dead) {
            dead = false;
            fresh = true;
        }

        const int action = std::min<int>((int)received, markov::ACTIONS - 1);
        received = 0;

        int emission;
        bool neural_emission = false;
#ifdef NANA_NEURAL_MODEL_DIR
        if (neural_opponent::forecast.active && neural_offset < 8) {
            emission = neural_opponent::forecast.sequences[neural_slot][neural_offset++];
            neural_emission = true;
            // Beyond eight pieces resume the existing HMM from its prior.
            fresh = true;
        } else
#endif
        {
            const auto& row = fresh ? p.initial : p.transition[action][jeopardy_bucket()][since][state];
            fresh = false;
            state = sample(row);
            emission = sample(p.emission[state]);
        }
        since = emission == 0 ? (uint8_t)std::min(since + 1, markov::SINCE - 1) : 0;
        if (emission == markov::DEATH) {
            dead = true;
#ifdef NANA_NEURAL_MODEL_DIR
            neural_terminal = neural_emission;
#endif
            deaths++;
            last_attack = 0;
            return 0;
        }

        pieces++;

        double attack = neural_emission ? emission : p.attack_value[emission];
        last_attack = attack;

        if (attack > 0) {
            const double cancelled = std::min<double>(attack, meter);
            meter -= (float)cancelled;
            attack -= cancelled;
        }
        else if (!p.stadium_rules) {
            meter -= std::min(meter, markov::GARBAGE_CAP);
        }

        total_damage += (float)attack;
        return attack;
    }

    void receiveAttack(double amount) {
        received += (float)amount;
        meter += (float)amount;
        if (markov::params().stadium_rules && last_attack <= 0) {
            meter = 0;
        }
    }

    int jeopardy_bucket() const {
        return markov::jeopardy_bucket(meter);
    }

    double jeopardy() const { return meter; }
    bool is_dead() const { return dead; }

    void reset_rng() { rng = fast_seed(); }
    void seed(uint32_t seed) { rng = seed; }

    inline void adopt(const RootOpponent& root);

    void resample_from_root() {
#ifdef NANA_NEURAL_MODEL_DIR
        neural_offset = 0;
        neural_terminal = false;
        if (neural_opponent::forecast.active) {
            neural_slot = (uint8_t)sample(neural_opponent::forecast.cdf);
            return;
        }
#endif
        if (markov::root_belief.active && !fresh) {
            state = (uint8_t)sample(markov::root_belief.cdf);
            since = (uint8_t)sample(markov::root_belief.since_cdf);
        }
    }

private:
#ifdef NANA_NEURAL_MODEL_DIR
    uint8_t neural_slot = 0;
    uint8_t neural_offset = 0;
    bool neural_terminal = false;
#endif
    uint8_t state = 0;
    uint8_t since = 0;
    bool fresh = true;
    float received = 0;
    uint32_t rng = fast_seed();

    float uniform() {
        rng = rng * 0x5d588b65 + 0x269ec3;
        return (float)(rng >> 8) * (1.0f / 16777216.0f);
    }

    template <size_t N>
    int sample(const std::array<float, N>& cdf) {
        const float u = uniform();
        int i = 0;
        while (i < (int)N - 1 && u >= cdf[i]) {
            i++;
        }
        return i;
    }
};

// Search without an opponent: no speculative garbage, no attack bookkeeping, no belief tracking. Code templated
// on the opponent compiles each of those blocks out under `if constexpr (Opp::simulated)`.
struct NoOpponent {
    static constexpr bool simulated = false;
};

struct RootOpponent {
    std::array<float, markov::STATES> state{};
    std::array<float, markov::SINCE> since{};
    bool fresh = true;
    float meter = 0;
    double last_attack = 0;

    void publish() const {
        markov::root_belief.active = !fresh;
        toCdf(state, markov::root_belief.cdf);
        toCdf(since, markov::root_belief.since_cdf);
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

inline void MarkovOpponent::adopt(const RootOpponent& root) {
    meter = root.meter;
    last_attack = root.last_attack;
    since = 0;
    received = 0;
    dead = false;
    fresh = root.fresh;
    state = 0;
}
