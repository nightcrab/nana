#pragma once
#include "Util/FastSeed.hpp"

#include <algorithm>
#include <array>
#include <cstdint>

// the opponent as an input-output hidden Markov model. we only see the opponent through the damage
// it sends, and it only sees us through the damage we send, so:
//   hidden state  z, the opponent's mode (the fit decides what the modes mean)
//   input         the damage we sent it since its last piece, binned
//   observed      jeopardy (cold clear's name for it): the garbage in its meter, bucketed. the meter
//                 is a function of the damage sent each way. garbage it tanks leaves jeopardy: once
//                 it's on the board, tangled up with its placements, it's part of the hidden state
//   observed      pieces since its last attack, bucketed. a function of its attacks, so the
//                 transition can depend on it: the chance of attacking again rises as it rebuilds,
//                 which lets the time between attacks take any distribution, not only geometric
//   transition    P(z' | z, input, jeopardy, since)
//   emission      P(attack | z'): its attack before cancelling, binned, or death
// the parameters are fitted by scripts/train_markov_opponent.py, which writes MarkovOpponentParams.hpp
namespace markov {
constexpr int STATES = 8;
// damage we send: 0, 1, ..., ACTIONS - 2, and ACTIONS - 1 or more
constexpr int ACTIONS = 6;
// buckets of the garbage in the opponent's meter, split at Params::jeopardy_limits
constexpr int JEOPARDY = 3;
// pieces since the opponent's last attack (or death): 0, 1, ..., SINCE - 2, and SINCE - 1 or more
constexpr int SINCE = 10;
// attack: 0, 1, ..., ATTACKS - 2, and ATTACKS - 1 or more (sent as Params::attack_value)
constexpr int ATTACKS = 8;
// the last emission is death
constexpr int DEATH = ATTACKS;
constexpr int EMISSIONS = ATTACKS + 1;
// garbage tanked at once, as in Opponent::process_meter
constexpr float GARBAGE_CAP = 8;

struct Params {
    std::array<float, STATES> initial;
    // transition[action][jeopardy][since][from][to]
    std::array<std::array<std::array<std::array<std::array<float, STATES>, STATES>, SINCE>, JEOPARDY>, ACTIONS> transition;
    std::array<std::array<float, EMISSIONS>, STATES> emission;
    std::array<float, ATTACKS> attack_value;
    // jeopardy bucket b holds a meter in [jeopardy_limits[b - 1], jeopardy_limits[b])
    std::array<float, JEOPARDY - 1> jeopardy_limits;
};

// the fitted parameters, with each distribution stored as a CDF for sampling
const Params& params();
}

class MarkovOpponent {
public:
    MarkovOpponent() = default;

    // the opponent's garbage meter: damage we sent that it hasn't cancelled or tanked yet
    float meter = 0;

    bool dead = false;
    double deaths = 0;
    float pieces = 0;
    float total_damage = 0;
    // attack this piece before cancelling, for logging
    double last_attack = 0;

    // Advances internal state and returns outgoing damage this turn
    double play() {
        const markov::Params& p = markov::params();

        if (dead) {
            // revive with a clean board, as Opponent::play does; the meter stays
            dead = false;
            fresh = true;
        }

        const int action = std::min<int>((int)received, markov::ACTIONS - 1);
        received = 0;

        if (fresh) {
            state = sample(p.initial);
            fresh = false;
        }
        else {
            state = sample(p.transition[action][jeopardy_bucket()][since][state]);
        }

        const int emission = sample(p.emission[state]);
        // any attack bin but 0 is an attack; a death starts it rebuilding too
        since = emission == 0 ? (uint8_t)std::min(since + 1, markov::SINCE - 1) : 0;
        if (emission == markov::DEATH) {
            dead = true;
            deaths++;
            last_attack = 0;
            return 0;
        }

        pieces++;

        double attack = p.attack_value[emission];
        last_attack = attack;

        if (attack > 0) {
            // cancel
            const double cancelled = std::min<double>(attack, meter);
            meter -= (float)cancelled;
            attack -= cancelled;
        }
        else {
            // tank
            meter -= std::min(meter, markov::GARBAGE_CAP);
        }

        total_damage += (float)attack;
        return attack;
    }

    void receiveAttack(double amount) {
        received += (float)amount;
        meter += (float)amount;
    }

    int jeopardy_bucket() const {
        const markov::Params& p = markov::params();
        int bucket = 0;
        while (bucket < markov::JEOPARDY - 1 && meter >= p.jeopardy_limits[bucket]) {
            bucket++;
        }
        return bucket;
    }

    // jeopardy: the garbage in the meter, not yet cancelled or tanked
    double jeopardy() const { return meter; }
    bool is_dead() const { return dead; }

    void reset_rng() { rng = fast_seed(); }
    void seed(uint32_t seed) { rng = seed; }

private:
    uint8_t state = 0;
    // pieces since its last attack, bucketed
    uint8_t since = 0;
    // the next piece draws its state from the initial distribution
    bool fresh = true;
    // damage received since the last piece
    float received = 0;
    uint32_t rng = fast_seed();

    // same LCG as RNG::getRand
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
