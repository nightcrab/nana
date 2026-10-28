#pragma once
#include "Util/FastSeed.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

// An opponent model only predicts what the opponent does each piece. EmulationGame applies the garbage rules
// (cancelling, the meter, deaths) and passes the model the inputs it needs.
//
// A model provides:
//   Prediction predict(float received, float meter, Rng&)   the next piece's attack, given the garbage the
//                                                           opponent received since its last piece and its meter
//   bool respawn()                                          after a death: true if the opponent plays on
//   void adopt(const Root&)                                 take the search root
//   void sample_root(Rng&)                                  draw this path's hidden state at the root
namespace opponent {

struct Prediction {
    float attack = 0;
    bool dies = false;
};

struct Rng {
    uint32_t state = fast_seed();

    void seed(uint32_t seed) { state = seed; }
    void reseed() { state = fast_seed(); }

    float uniform() {
        state = state * 0x5d588b65 + 0x269ec3;
        return (float)(state >> 8) * (1.0f / 16777216.0f);
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
struct None {
    static constexpr bool simulated = false;
};

}
