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
// garbage tanked at once (8 lines, outside stadium rules)
constexpr float GARBAGE_CAP = 8;

struct Params {
    std::array<float, STATES> initial;
    // transition[action][jeopardy][since][from][to]
    std::array<std::array<std::array<std::array<std::array<float, STATES>, STATES>, SINCE>, JEOPARDY>, ACTIONS> transition;
    std::array<std::array<float, EMISSIONS>, STATES> emission;
    std::array<float, ATTACKS> attack_value;
    // jeopardy bucket b holds a meter in [jeopardy_limits[b - 1], jeopardy_limits[b])
    std::array<float, JEOPARDY - 1> jeopardy_limits;
    // death_value[state][jeopardy][since]: the value of the opponent's death from here, either P(it dies
    // within its next `horizon` pieces) or E[discount^(pieces until it dies)],
    // given the state of its last piece and the jeopardy and since buckets before its next. fitted
    // from games, so it averages over how the other player really attacked
    std::array<std::array<std::array<float, SINCE>, JEOPARDY>, STATES> death_value;
    // 0 when discounted
    int horizon;
    // 0 for a fixed horizon
    float discount;
    // fitted on stadium games: a piece that clears no lines tanks the whole meter when the other
    // player's attack of the same turn arrives, so jeopardy is what's left of the meter after that.
    // otherwise (nana_emulate logs), tank 8 lines on a piece that sends nothing
    bool stadium_rules;
};

// the fitted parameters, with each distribution stored as a CDF for sampling
const Params& params();
// death_value for an opponent that hasn't drawn its state yet: averaged over the initial distribution
float fresh_death_value(int jeopardy, int since);

// the jeopardy bucket of a meter
inline int jeopardy_bucket(float meter) {
    const Params& p = params();
    int bucket = 0;
    while (bucket < JEOPARDY - 1 && meter >= p.jeopardy_limits[bucket]) {
        bucket++;
    }
    return bucket;
}

// the probability of entry i of a distribution stored as a CDF
template <size_t N>
inline float probability(const std::array<float, N>& cdf, int i) {
    return cdf[i] - (i > 0 ? cdf[i - 1] : 0.0f);
}

// the search's belief over the opponent's hidden state at the root, as a CDF: every simulation that
// starts from the root draws its opponent's state from it (MarkovOpponent::resample_from_root). the
// main thread writes it before a search starts and nothing writes it during one, so it sits read-only
// in every core's cache; its own cache line keeps writes to neighbouring data from invalidating it
struct alignas(64) RootBelief {
    std::array<float, STATES> cdf{};
    // its pieces-since-its-last-attack counter, which the search's root doesn't observe either
    std::array<float, SINCE> since_cdf{};
    bool active = false;
};
inline RootBelief root_belief;
}

struct RootOpponent;

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

    // the search sets this: instead of sampling its death, each piece returns the probability it
    // died (last_death_prob) and plays on as if it survived, so the search can count the kill as an
    // expectation rather than a rare sampled event
    bool expected_death = false;
    float last_death_prob = 0;

    // Advances internal state and returns outgoing damage this turn
    double play() {
        const markov::Params& p = markov::params();

        if (dead) {
            // revive with a clean board; the meter stays
            dead = false;
            fresh = true;
        }

        const int action = std::min<int>((int)received, markov::ACTIONS - 1);
        received = 0;

        const auto& row = fresh ? p.initial : p.transition[action][jeopardy_bucket()][since][state];
        fresh = false;
        int emission;
        if (expected_death) {
            emission = play_surviving(p, row);
        }
        else {
            state = sample(row);
            emission = sample(p.emission[state]);
        }
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
        else if (!p.stadium_rules) {
            // tank
            meter -= std::min(meter, markov::GARBAGE_CAP);
        }

        total_damage += (float)attack;
        return attack;
    }

    void receiveAttack(double amount) {
        received += (float)amount;
        meter += (float)amount;
        // stadium rules: its last piece cleared no lines (we only see that it sent nothing), so it
        // tanks the whole meter this turn
        if (markov::params().stadium_rules && last_attack <= 0) {
            meter = 0;
        }
    }

    int jeopardy_bucket() const {
        return markov::jeopardy_bucket(meter);
    }

    // the value of the opponent's death from here (see Params::death_value); 1 if it just died
    float death_value() const {
        if (dead) {
            return 1.0f;
        }
        if (fresh) {
            return markov::fresh_death_value(jeopardy_bucket(), since);
        }
        return markov::params().death_value[state][jeopardy_bucket()][since];
    }

    // jeopardy: the garbage in the meter, not yet cancelled or tanked
    double jeopardy() const { return meter; }
    bool is_dead() const { return dead; }

    void reset_rng() { rng = fast_seed(); }
    void seed(uint32_t seed) { rng = seed; }

    // take what's known of the real opponent at the search root: its meter and whether it has played
    // yet. its hidden state and since counter come from the root belief (resample_from_root)
    inline void adopt(const RootOpponent& root);

    // draw its hidden state and since counter from the root belief, if the search has one
    void resample_from_root() {
        if (markov::root_belief.active && !fresh) {
            state = (uint8_t)sample(markov::root_belief.cdf);
            since = (uint8_t)sample(markov::root_belief.since_cdf);
        }
    }

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
    int sample(const std::array<float, N>& cdf, float scale = 1.0f) {
        const float u = uniform() * scale;
        int i = 0;
        while (i < (int)N - 1 && u >= cdf[i]) {
            i++;
        }
        return i;
    }

    // expected_death: sets last_death_prob to P(it dies this piece) and state to a draw of its next
    // state given that it doesn't, and returns an emission drawn from that state without death
    int play_surviving(const markov::Params& p, const std::array<float, markov::STATES>& row) {
        std::array<float, markov::STATES> survive;
        float death = 0;
        float total = 0;
        float previous = 0;
        for (int j = 0; j < markov::STATES; j++) {
            const float to = row[j] - previous;
            previous = row[j];
            // death is the last emission, so its probability is what the attacks leave
            const float alive = p.emission[j][markov::DEATH - 1];
            death += to * (1.0f - alive);
            total += to * alive;
            survive[j] = total;
        }
        last_death_prob = death;
        if (total <= 0) {
            // it dies from every state; play on from any
            state = sample(row);
            return 0;
        }
        state = sample(survive, total);
        return sample(p.emission[state], p.emission[state][markov::DEATH - 1]);
    }
};

// what's known of the real opponent at a search root, from whoever can see it (tbp: its board, through
// the opponent encoder). the search's HMM opponent carries on from it instead of starting fresh
struct RootOpponent {
    // P(its hidden state) and P(its since counter), meaningful unless fresh
    std::array<float, markov::STATES> state{};
    std::array<float, markov::SINCE> since{};
    // it hasn't played yet (an empty board): its first state comes from the initial distribution
    bool fresh = true;
    // garbage pending against it, if known
    float meter = 0;
    // its last piece's attack, if known (stadium rules tank on a piece that sent nothing)
    double last_attack = 0;

    // publish as the root belief for the next search (main thread, between searches)
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
