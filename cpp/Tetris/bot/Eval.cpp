#include "Eval.hpp"
#include "Config.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "engine/Board.hpp"
#include "engine/Game.hpp"
#include "Move.hpp"

static bool Eval::is_top_quarter(const Board& board) {
    // do NOT touch anything this vectorizes
    constexpr uint32_t top_quarter_collider = ~((1 << 15) - 1);

    bool ret = false;

    for (size_t x = 0; x < Board::width; x++) {
        if (board.get_column(x) & top_quarter_collider)
            ret = true;
    }
    return ret;
}

static bool Eval::is_top_half(const Board& board) {
    // do NOT touch anything this vectorizes
    constexpr uint32_t top_half_collider = ~((1 << 10) - 1);

    bool ret = false;

    for (size_t x = 0; x < Board::width; x++) {
        if (board.get_column(x) & top_half_collider)
            ret = true;
    }
    return ret;
}


static bool is_low(const Board& board) {
    constexpr uint32_t high_collider = ~((1 << 4) - 1);

    bool ret = true;

    for (size_t x = 0; x < Board::width; x++) {
        if (board.get_column(x) & high_collider)
            ret = false;
    }
    return ret;
}

static bool is_perfect_clear(const Board& board) {

    bool ret = true;

    for (size_t x = 0; x < Board::width; x++) {
        if (board.get_column(x))
            ret = false;
    }
    return ret;
}

std::pair<int, int> Eval::cavities_overhangs(const Board& board) {
    int cavities = 0;
    int overhangs = 0;

    for (int i = 0; i < Board::width; ++i) {
        auto col = board.board[i];
        col = ~col;
        col = col << std::countl_one(col);
        cavities += (board.board[i] != 0) * std::popcount(col);
    }

    for (int i = 1; i < Board::width; ++i) {
        auto col1 = board.board[i - 1];
        auto col2 = board.board[i];
        col1 = col1 >> (32 - std::countl_zero(col2));
        auto col3 = col1;
        col1 = ~col1;
        col1 = col1 << std::countl_one(col1);
        overhangs += (col3 != 0) * std::popcount(col1);
    }

    return { cavities, overhangs };
}

int Eval::well_position(const Board& board) {

    int max_air = -1;
    int well_position = 0;

    for (int i = 0; i < Board::width; ++i) {
        auto& col = board.board[i];
        int air = std::countl_zero(col);
        if (air > max_air) {
            max_air = air;
            well_position = i;
        }
    }

    return well_position;
}

// lowest height, highest height
std::pair<int, int> Eval::height_features(const Board& board) {

    // air is the complement of height

    int max_air = -1;
    int min_air = 1 << 30;

    for (int i = 0; i < Board::width; ++i) {
        auto& col = board.board[i];
        int air = std::countl_zero(col);
        max_air = std::max(air, max_air);
        min_air = std::min(air, min_air);
    }

    return { 32 - max_air, 32 - min_air };
}

std::pair<int, int> Eval::n_covered_cells(Board board) {
    int covered = 0;
    int covered_sq = 0;

    for (int i = 0; i < Board::width; ++i) {
        auto col = board.board[i];
        int col_covered = 0;
        int zeros = std::countl_zero(col);
        col <<= zeros; // 00011101 -> 11101
        int ones = std::countl_one(col);
        if (zeros + ones != 32) {
            col_covered += col_covered + ones;
        }
        covered += col_covered;
        covered_sq += col_covered * col_covered;
    }

    return { covered, covered_sq };
}

static auto abss(auto num) {
    if (num < 0)
        return -num;
    else
        return num;

}

static std::pair<int, int> Eval::get_bumpiness(const Board& board) {
    int bumpiness = 0;
    int bumpiness_sq = 0;

    std::array<int, Board::width> air;

    for (int i = 0; i < Board::width; ++i) {
        air[i] = std::countl_zero(board.board[i]);
    }

    std::array<int, Board::width> bump;


    for (int i = 1; i < Board::width; ++i) {
        bump[i] = abss(air[i] - air[i - 1]);
    }

    for (int i = 1; i < Board::width; ++i) {
        bumpiness += bump[i];
        bumpiness_sq += bump[i] * bump[i];

    }
    return { bumpiness, bumpiness_sq };
}

// Identify clean count to 4
static bool Eval::ct4(const Board& board) {

    int garbage_height = height_features(board).first;

    bool quad_ready = false;

    for (int i = 0; i < Board::width; ++i) {
        auto& col = board.board[i];
        quad_ready = quad_ready && ((((col >> garbage_height) | 0b1111) & 0b1111) == 0b1111);
    }

    if (!quad_ready) {
        return false;
    }

    if (garbage_height == 0) {
        return true;
    }

    for (int i = 0; i < Board::width; ++i) {
        if (!board.get(i, garbage_height - 1)) {
            // check we have exactly 4 rows filled above this hole
            if (std::countl_zero(board.board[i]) != 28 - garbage_height) {
                return false;
            }
        }
    }

    return true;
}

int Eval::get_row_transitions(const Board& board) {
    int transitions = 0;
    for (int i = 0; i < Board::width - 1; i += 2) {
        auto& col1 = board.board[i];
        auto& col2 = board.board[i + 1];
        transitions += std::popcount(col1 ^ col2);
    }
    return transitions;
}

// Convex: the no-spin style, leaving out the features that only score boards with an overhang or cavity above
// the lowest column (overhangs, T-spins, wasted T). evalConvex scores those boards on their own.
template <bool Convex>
static double evalBoard(const Board& board, int lines, bool tspin, bool waste_t) {

    static const NanaConfig& c = config();

    double score = 0.0;

    std::pair<int, int> values;

    if (Eval::is_top_half(board))
        score += c.top_half;

    if (Eval::is_top_quarter(board))
        score += c.top_quarter;

    if (is_low(board))
        score += c.low;

    if (is_perfect_clear(board))
        score += c.perfect_clear;


    values = Eval::cavities_overhangs(board);

    score += values.first * c.cavity_cells;

    score += values.first * values.first * c.cavity_cells_sq;

    if constexpr (!Convex) {
        score += values.second * c.overhangs;

        score += values.second * values.second * c.overhangs_sq;
    }

    values = Eval::height_features(board);

    score += values.second * c.height;

    if (Eval::ct4(board)) {
        score += c.counting;
    }

    values = Eval::n_covered_cells(board);

    score += values.first * c.covered_cells;
    score += values.second * c.covered_cells_sq;

    values = Eval::get_bumpiness(board);

    score += values.first * c.bumpiness;
    score += values.second * c.bumpiness_sq;

    score += c.well_columns[Eval::well_position(board)];
    score += c.clears[lines];

    score += Eval::get_row_transitions(board) * c.row_transitions;

    if (!Convex && tspin && lines < (int)std::size(c.tspins)) {
        score += c.tspins[lines];
    }

    if (!Convex && waste_t) {
        score += c.wasted_t;
    }

    return (score + 10000) / 10000;
}

double Eval::nanaBoard(const Board& board, int lines, bool tspin, bool waste_t) {
    return evalBoard<false>(board, lines, tspin, waste_t);
}

double Eval::evalNana(const Game& game, const Piece& piece, double* move_part) {
    // only the board changes, so only the board is copied
    Board board = game.board;
    board.set(piece);
    int lines = board.clearLines();
    spinType spin = piece.spin;
    bool tspin = false;
    if (spin == spinType::normal &&
        (piece.type == PieceType::T || !std::holds_alternative<TetrioS1>(game.mode))) {
        tspin = true;
    }

    bool waste_t = false;

    if (piece.type == PieceType::T) {
        if (!tspin) {
            waste_t = true;
        }
    }

    if (move_part) {
        static const NanaConfig& c = config();
        double score = c.clears[lines];
        if (tspin && lines < (int)std::size(c.tspins)) {
            score += c.tspins[lines];
        }
        if (waste_t) {
            score += c.wasted_t;
        }
        *move_part = score / 10000;
    }
    return nanaBoard(board, lines, tspin, waste_t);
}

double Eval::evalConvex(const Game& game, const Piece& piece, double* move_part, bool* convex) {
    static const NanaConfig& c = config();
    // death is worth 0, so every live board stays at least one point above it
    constexpr double alive = 1.0 / 10000;
    // only the board changes, so only the board is copied
    Board board = game.board;
    board.set(piece);
    int lines = board.clearLines();
    const bool is_convex = board.surface_convex();
    if (convex) {
        *convex = is_convex;
    }
    if (!is_convex) {
        if (move_part) {
            *move_part = 0;
        }
        return std::max((c.nonconvex + 10000) / 10000, alive);
    }
    if (move_part) {
        *move_part = c.clears[lines] / 10000;
    }
    return std::max(evalBoard<true>(board, lines, false, false), alive);
}

double Eval::evaluate(const Context& context, const Piece& piece, double* move_part, bool* convex) {
    switch (context.mode) {
    case EvalMode::CC:
        return evalCC(context.game, piece, context.bag_left, move_part, context.belief, context.meter);
    case EvalMode::Convex:
        return evalConvex(context.game, piece, move_part, convex);
    default:
        return evalNana(context.game, piece, move_part);
    }
}
