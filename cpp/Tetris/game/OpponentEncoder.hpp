#pragma once
#include "MarkovOpponent.hpp"
#include "engine/Board.hpp"

#include <array>

namespace markov {
constexpr int ENCODER_FEATURES = 11;

struct EncoderParams {
    double fingerprint;
    std::array<float, ENCODER_FEATURES> mean;
    std::array<float, ENCODER_FEATURES> scale;
    std::array<std::array<float, STATES>, ENCODER_FEATURES> state_weights;
    std::array<float, STATES> state_bias;
    std::array<std::array<float, SINCE>, ENCODER_FEATURES> since_weights;
    std::array<float, SINCE> since_bias;
    std::array<float, STATES> state_prior;
    std::array<float, SINCE> since_prior;
};

std::array<float, ENCODER_FEATURES> boardFeatures(const Board& board, int combo, int b2b);

double hmmFingerprint();

bool encoderAvailable();

bool encodeOpponent(const Board& board, int combo, int b2b, std::array<float, STATES>& state,
                    std::array<float, SINCE>& since);

bool opponentEvidence(const Board& board, int combo, int b2b, std::array<float, STATES>& state,
                      std::array<float, SINCE>& since);
}
