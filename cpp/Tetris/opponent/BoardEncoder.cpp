#include "BoardEncoder.hpp"
#include "IOHMMParams.hpp"
#include "BoardEncoderParams.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace opponent {

std::array<float, BOARD_FEATURES> board_features(const Board& board, int combo, int b2b) {
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

double iohmm_fingerprint() {
    double fingerprint = 0;
    int i = 0;
    for (const auto& row : fitted_iohmm.emission) {
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

bool board_encoder_available() {
    static const bool available = [] {
        if (fitted_board_encoder.fingerprint == 0) {
            return false;
        }
        if (std::abs(fitted_board_encoder.fingerprint - iohmm_fingerprint()) > 1e-3) {
            std::cerr << "warning: the opponent encoder was trained against another opponent HMM; it's off" << std::endl;
            return false;
        }
        return true;
    }();
    return available;
}

namespace {
template <size_t N>
void softmaxHead(const std::array<float, BOARD_FEATURES>& x, const std::array<std::array<float, N>, BOARD_FEATURES>& w,
                 const std::array<float, N>& b, std::array<float, N>& out) {
    std::array<float, N> logits = b;
    for (int f = 0; f < BOARD_FEATURES; f++) {
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

bool encode_board(const Board& board, int combo, int b2b, std::array<float, STATES>& state,
                    std::array<float, SINCE>& since) {
    if (!board_encoder_available()) {
        return false;
    }
    const auto features = board_features(board, combo, b2b);
    std::array<float, BOARD_FEATURES> x;
    for (int f = 0; f < BOARD_FEATURES; f++) {
        x[f] = (features[f] - fitted_board_encoder.mean[f]) / fitted_board_encoder.scale[f];
    }
    softmaxHead(x, fitted_board_encoder.state_weights, fitted_board_encoder.state_bias, state);
    softmaxHead(x, fitted_board_encoder.since_weights, fitted_board_encoder.since_bias, since);
    return true;
}

bool board_evidence(const Board& board, int combo, int b2b, std::array<float, STATES>& state,
                      std::array<float, SINCE>& since) {
    if (!encode_board(board, combo, b2b, state, since)) {
        return false;
    }
    for (int z = 0; z < STATES; z++) {
        state[z] /= std::max(fitted_board_encoder.state_prior[z], 1e-6f);
    }
    for (int s = 0; s < SINCE; s++) {
        since[s] /= std::max(fitted_board_encoder.since_prior[s], 1e-6f);
    }
    return true;
}

}
