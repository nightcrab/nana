#include "OpponentEncoder.hpp"
#include "MarkovOpponentParams.hpp"
#include "OpponentEncoderParams.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace markov {

std::array<float, ENCODER_FEATURES> boardFeatures(const Board& board, int combo, int b2b) {
    constexpr uint32_t rows = (1u << 20) - 1;
    std::array<uint32_t, Board::width> col;
    std::array<int, Board::width> height;
    for (int x = 0; x < Board::width; x++) {
        col[x] = board.board[x] & rows;
        height[x] = col[x] ? 32 - std::countl_zero(col[x]) : 0;
    }

    int max_height = 0;
    int min_height = 99;
    int sum_height = 0;
    int holes = 0;
    int covered = 0;
    for (int x = 0; x < Board::width; x++) {
        const int h = height[x];
        max_height = std::max(max_height, h);
        min_height = std::min(min_height, h);
        sum_height += h;
        const int filled = std::popcount(col[x]);
        holes += h - filled;
        if (h > filled) {
            // filled cells from the lowest hole up
            const int lowest_hole = std::countr_one(col[x]);
            covered += std::popcount(col[x] >> lowest_hole);
        }
    }
    int bumpiness = 0;
    int row_transitions = 0;
    for (int x = 0; x + 1 < Board::width; x++) {
        bumpiness += std::abs(height[x] - height[x + 1]);
        row_transitions += std::popcount((col[x] ^ col[x + 1]) & rows);
    }
    int max_well = 0;
    for (int x = 0; x < Board::width; x++) {
        const int left = x > 0 ? height[x - 1] : 99;
        const int right = x + 1 < Board::width ? height[x + 1] : 99;
        max_well = std::max(max_well, std::min(left, right) - height[x]);
    }
    return { (float)max_height, (float)sum_height / Board::width, (float)min_height, (float)holes, (float)covered,
             (float)bumpiness, (float)max_well, (float)row_transitions, (float)combo, (float)b2b,
             (float)(max_height >= 15) };
}

double hmmFingerprint() {
    double fingerprint = 0;
    int i = 0;
    for (const auto& row : fitted_params.emission) {
        double total = 0;
        for (float p : row) {
            total += p;
        }
        for (float p : row) {
            fingerprint += (++i) * (p / total);
        }
    }
    return fingerprint;
}

bool encoderAvailable() {
    static const bool available = [] {
        if (fitted_encoder.fingerprint == 0) {
            return false;
        }
        if (std::abs(fitted_encoder.fingerprint - hmmFingerprint()) > 1e-3) {
            std::cerr << "warning: the opponent encoder was trained against another opponent HMM; it's off until "
                         "scripts/board_encoder.py retrains it" << std::endl;
            return false;
        }
        return true;
    }();
    return available;
}

namespace {
template <size_t N>
void softmaxHead(const std::array<float, ENCODER_FEATURES>& x, const std::array<std::array<float, N>, ENCODER_FEATURES>& w,
                 const std::array<float, N>& b, std::array<float, N>& out) {
    std::array<float, N> logits = b;
    for (int f = 0; f < ENCODER_FEATURES; f++) {
        for (size_t k = 0; k < N; k++) {
            logits[k] += x[f] * w[f][k];
        }
    }
    const float top = *std::max_element(logits.begin(), logits.end());
    float total = 0;
    for (size_t k = 0; k < N; k++) {
        out[k] = std::exp(logits[k] - top);
        total += out[k];
    }
    for (float& p : out) {
        p /= total;
    }
}
}

bool encodeOpponent(const Board& board, int combo, int b2b, std::array<float, STATES>& state,
                    std::array<float, SINCE>& since) {
    if (!encoderAvailable()) {
        return false;
    }
    const auto features = boardFeatures(board, combo, b2b);
    std::array<float, ENCODER_FEATURES> x;
    for (int f = 0; f < ENCODER_FEATURES; f++) {
        x[f] = (features[f] - fitted_encoder.mean[f]) / fitted_encoder.scale[f];
    }
    softmaxHead(x, fitted_encoder.state_weights, fitted_encoder.state_bias, state);
    softmaxHead(x, fitted_encoder.since_weights, fitted_encoder.since_bias, since);
    return true;
}

bool opponentEvidence(const Board& board, int combo, int b2b, std::array<float, STATES>& state,
                      std::array<float, SINCE>& since) {
    if (!encodeOpponent(board, combo, b2b, state, since)) {
        return false;
    }
    for (int z = 0; z < STATES; z++) {
        state[z] /= std::max(fitted_encoder.state_prior[z], 1e-6f);
    }
    for (int s = 0; s < SINCE; s++) {
        since[s] /= std::max(fitted_encoder.since_prior[s], 1e-6f);
    }
    return true;
}

}
