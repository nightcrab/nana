#include "MarkovOpponent.hpp"
#include "MarkovOpponentParams.hpp"

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

static Params makeCdfs() {
    Params p = fitted_params;
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
