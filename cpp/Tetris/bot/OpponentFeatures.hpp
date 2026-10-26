#pragma once
#include "engine/Board.hpp"
#include "engine/Piece.hpp"

#include <array>
#include <optional>

// What the encoder sees of the opponent at one position (TBP2 shows all of it), and the features it is
// computed into. The training data (scripts/train_encoder_hmm.py, via nana_opp_features) and the bot use
// this one function, so they can't drift apart.
namespace opponent {

constexpr int PREVIEWS = 5;
// placements in the rollout: the current piece and every preview
constexpr int ROLLOUT = 1 + PREVIEWS;

struct View {
    Board board{};
    PieceType current = PieceType::Empty;
    std::optional<PieceType> hold;
    std::array<PieceType, PREVIEWS> queue{ PieceType::Empty, PieceType::Empty, PieceType::Empty, PieceType::Empty,
                                           PieceType::Empty };
    int combo = 0;
    int b2b = 0;
    // garbage waiting to land on the opponent
    int meter = 0;
};

enum Feature {
    HEIGHT,
    WELL_DEPTH,
    ROW_TRANSITIONS,
    BUMPINESS,
    CAVITIES,
    OVERHANGS,
    COVERED,
    TSLOT,
    TSLOT_LINES,
    COMBO,
    B2B,
    METER,
    // the most attack any placement of the current or hold piece sends now
    BEST_ATTACK_NOW,
    // attack sent by each placement of a greedy rollout with Cold Clear's default eval over the known queue
    ROLLOUT_ATTACK,
    ROLLOUT_DEAD = ROLLOUT_ATTACK + ROLLOUT,
    FEATURES
};

const char* featureName(int feature);

std::array<float, FEATURES> features(const View& view);

}
