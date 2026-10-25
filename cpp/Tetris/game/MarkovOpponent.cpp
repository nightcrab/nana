#include "MarkovOpponent.hpp"
#include "MarkovOpponentParams.hpp"

#include <cmath>
#include <cstdlib>

namespace markov {

template <size_t N>
static void toCdf(std::array<float, N>& row) {
    float total = 0;
    for (float p : row) {
        total += p;
    }
    float sum = 0;
    for (float& p : row) {
        sum += p / total;
        p = sum;
    }
    // rounding must not leave a gap above the last entry
    row[N - 1] = 1.0f;
}

// the model in use is the fit sharpened: every transition row and the attack part of every emission
// row become p^a renormalised (a = 1: the fit; large a: nearly the argmax), the death probability
// kept as fitted. the default is a = 4; a = 2 gives about 0.45 bits/piece of 5-piece attack entropy
// given the state, the encoder's target, and the raw fit (a = 1) is nearly uninformative.
// NANA_MARKOV_SHARPEN=a overrides it (1 for the raw fit). the game's opponent, the search and the
// filter all share params(), so all get the same model
static float sharpenExponent() {
    const char* text = std::getenv("NANA_MARKOV_SHARPEN");
    return text && *text ? (float)std::atof(text) : 4.0f;
}

template <size_t N>
static void sharpen(std::array<float, N>& row, float a, size_t n = N) {
    float before = 0;
    float after = 0;
    for (size_t i = 0; i < n; i++) {
        before += row[i];
        row[i] = row[i] > 0 ? std::pow(row[i], a) : 0.0f;
        after += row[i];
    }
    if (after <= 0) {
        return;
    }
    for (size_t i = 0; i < n; i++) {
        row[i] *= before / after;
    }
}

static Params makeCdfs() {
    Params p = fitted_params;
    const float a = sharpenExponent();
    if (a != 1.0f) {
        for (auto& by_jeopardy : p.transition) {
            for (auto& by_since : by_jeopardy) {
                for (auto& rows : by_since) {
                    for (auto& row : rows) {
                        sharpen(row, a);
                    }
                }
            }
        }
        for (auto& row : p.emission) {
            // the attack bins only, keeping their total: death stays as likely as fitted
            sharpen(row, a, ATTACKS);
        }
    }
    toCdf(p.initial);
    for (auto& by_jeopardy : p.transition) {
        for (auto& by_since : by_jeopardy) {
            for (auto& rows : by_since) {
                for (auto& row : rows) {
                    toCdf(row);
                }
            }
        }
    }
    for (auto& row : p.emission) {
        toCdf(row);
    }
    return p;
}

const Params& params() {
    static const Params cdfs = makeCdfs();
    return cdfs;
}

float fresh_death_value(int jeopardy, int since) {
    const Params& p = fitted_params;
    float total = 0;
    float weighted = 0;
    for (int z = 0; z < STATES; z++) {
        total += p.initial[z];
        weighted += p.initial[z] * p.death_value[z][jeopardy][since];
    }
    return total > 0 ? weighted / total : 0.0f;
}

}
