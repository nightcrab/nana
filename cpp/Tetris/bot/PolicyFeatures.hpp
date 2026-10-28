#pragma once
#include "Eval.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <stdexcept>
#include <string>

// Shared training/inference basis; version 1 of the fitted policy artifact.
namespace PolicyFeatures {
constexpr size_t FEATURES = 43;
using Features = std::array<double, FEATURES>;
inline const std::array<std::string, FEATURES> names = {
    "top_half", "top_quarter", "low", "cavity_cells", "cavity_cells_sq",
    "overhangs", "overhangs_sq", "covered_cells", "covered_cells_sq",
    "bumpiness", "bumpiness_sq", "height", "row_transitions", "counting",
    "perfect_clear", "wasted_t",
    "well_columns.0", "well_columns.1", "well_columns.2", "well_columns.3", "well_columns.4",
    "well_columns.5", "well_columns.6", "well_columns.7", "well_columns.8", "well_columns.9",
    "clears.0", "clears.1", "clears.2", "clears.3", "clears.4",
    "tspins.0", "tspins.1", "tspins.2", "tspins.3",
    "back_to_back", "b2b_clear", "combo_garbage", "move_time", "well_depth",
    "mini_tspins.0", "mini_tspins.1", "mini_tspins.2"
};
inline constexpr Features scales = {
    1,1,1,8,64,8,64,32,256,16,128,20,64,1,1,1,
    1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1, 1,1,1,1, 1,1,8,40,8,1,1,1
};

inline Features features(const Game& parent, const Piece& piece, int well_cap) {
    Game after = parent;
    const bool held = piece.type != parent.current_piece.type;
    after.place_piece(piece);
    const int lines = after.board.clearLines();
    if (lines > 4) throw std::runtime_error("placement clears more than four lines");
    const auto spin = std::holds_alternative<TetrioS1>(after.mode) && piece.type != PieceType::T
        ? spinType::null : piece.spin;
    const bool full = spin == spinType::normal;
    const bool pc = after.board.is_empty();
    const int attack = after.damage_sent(lines, spin, pc);
    const auto& board = after.board;
    const int height = Eval::height_features(board).second;
    const auto [holes, overhangs] = Eval::cavities_overhangs(board);
    const auto [covered, covered_sq] = Eval::n_covered_cells(board);
    int bump = 0, bump_sq = 0;
    for (size_t x = 1; x < Board::width; ++x) {
        const int d = std::abs(std::countl_zero(board.board[x]) - std::countl_zero(board.board[x-1]));
        bump += d;
        bump_sq += d*d;
    }
    Features f{};
    f[0] = height > 10; f[1] = height > 15; f[2] = height <= 4;
    f[3] = holes; f[4] = holes*holes; f[5] = overhangs; f[6] = overhangs*overhangs;
    f[7] = covered; f[8] = covered_sq; f[9] = bump; f[10] = bump_sq;
    f[11] = height; f[12] = Eval::get_row_transitions(board); f[13] = Eval::ct4(board);
    f[14] = pc; f[15] = piece.type == PieceType::T && !full;
    const int well = Eval::well_position(board);
    f[16 + well] = 1;
    f[26 + lines] = 1;
    if (full && lines < 4) f[31 + lines] = 1;
    f[35] = after.b2b > 0;
    f[36] = lines > 0 && (lines == 4 || spin != spinType::null) && parent.b2b > 0;
    f[37] = lines > 0 ? attack : 0;
    static constexpr int rotations[] = {0,1,2,1};
    f[38] = std::abs(int(piece.position.x)-4) + rotations[int(piece.rotation)] + held + (lines > 0 ? 40 : 0);
    f[39] = Eval::well_depth(board, well, well_cap);
    if (spin == spinType::mini && lines < 3) f[40 + lines] = 1;
    return f;
}

inline double logit(const Game& game, const Piece& piece, const Features& weights, int well_cap) {
    const auto raw = features(game, piece, well_cap);
    double score = 0;
    for (size_t k = 0; k < FEATURES; ++k)
        score += weights[k] * float(raw[k] / scales[k]);
    return score;
}
} // namespace PolicyFeatures
