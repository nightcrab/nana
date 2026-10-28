#pragma once
#include "IOHMM.hpp"
#include "engine/Board.hpp"

#include <array>

namespace opponent {
constexpr int BOARD_FEATURES = 11;

struct BoardEncoderParams {
    double fingerprint;
    std::array<float, BOARD_FEATURES> mean;
    std::array<float, BOARD_FEATURES> scale;
    std::array<std::array<float, STATES>, BOARD_FEATURES> state_weights;
    std::array<float, STATES> state_bias;
    std::array<std::array<float, SINCE>, BOARD_FEATURES> since_weights;
    std::array<float, SINCE> since_bias;
    std::array<float, STATES> state_prior;
    std::array<float, SINCE> since_prior;
};

std::array<float, BOARD_FEATURES> board_features(const Board& board, int combo, int b2b);

double iohmm_fingerprint();

bool board_encoder_available();

bool encode_board(const Board& board, int combo, int b2b, std::array<float, STATES>& state,
                    std::array<float, SINCE>& since);

bool board_evidence(const Board& board, int combo, int b2b, std::array<float, STATES>& state,
                      std::array<float, SINCE>& since);
}
