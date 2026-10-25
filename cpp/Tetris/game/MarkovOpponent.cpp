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
    row[N - 1] = 1.0f;
}

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
