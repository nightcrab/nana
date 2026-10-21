#pragma once
#include "MarkovOpponent.hpp"
#include "engine/Board.hpp"

#include <array>

// the opponent's board -> what the search's root needs to know of it: a belief over the opponent HMM's
// hidden state, and over its pieces-since-its-last-attack counter (an input to the HMM that the board
// shows through attack readiness: a ready well, a spin slot, a stack built up since its last clear).
// two softmax regressions on a few board features, fitted by scripts/board_encoder.py on stadium games
// (the state head on the HMM's smoothed posteriors, the since head on the logged counters), which
// writes OpponentEncoderParams.hpp. the HMM's states are only defined by one fit, so the encoder
// carries a fingerprint of the HMM it was trained against: after a refit it no longer matches and is
// switched off until it's retrained
namespace markov {
// max height, mean height, lowest column, holes, covered cells, bumpiness, deepest well, row
// transitions, combo, b2b, stack in the top quarter
constexpr int ENCODER_FEATURES = 11;

struct EncoderParams {
    // of the HMM it was trained against (see hmmFingerprint); 0 = no encoder
    double fingerprint;
    std::array<float, ENCODER_FEATURES> mean;
    std::array<float, ENCODER_FEATURES> scale;
    std::array<std::array<float, STATES>, ENCODER_FEATURES> state_weights;
    std::array<float, STATES> state_bias;
    std::array<std::array<float, SINCE>, ENCODER_FEATURES> since_weights;
    std::array<float, SINCE> since_bias;
    // the heads' average outputs over the training data: dividing by these turns an output into
    // evidence to multiply into another belief (OpponentFilter::root) without counting the prior twice
    std::array<float, STATES> state_prior;
    std::array<float, SINCE> since_prior;
};

// the features, as scripts/board_encoder.py computes them (the lowest 20 rows of the board)
std::array<float, ENCODER_FEATURES> boardFeatures(const Board& board, int combo, int b2b);

// a checksum of the fitted HMM's emissions, to tell whether an encoder was trained against it
double hmmFingerprint();

// whether a usable encoder is built in (trained against this HMM)
bool encoderAvailable();

// P(hidden state | board, combo, b2b) and P(since counter | ...); false (outputs untouched) if there's
// no usable encoder
bool encodeOpponent(const Board& board, int combo, int b2b, std::array<float, STATES>& state,
                    std::array<float, SINCE>& since);

// the same as evidence: the outputs divided by their training averages; false if no usable encoder
bool opponentEvidence(const Board& board, int combo, int b2b, std::array<float, STATES>& state,
                      std::array<float, SINCE>& since);
}
