#include "Eval.hpp"

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

namespace eval_constants {
    [[maybe_unused]] constexpr auto top_half = -130.0;
    [[maybe_unused]] constexpr auto top_quarter = -499.0;
    [[maybe_unused]] constexpr auto low = -50.0;
    [[maybe_unused]] constexpr auto cavity_cells = -176.0;
    [[maybe_unused]] constexpr auto cavity_cells_sq = -6.0;
    [[maybe_unused]] constexpr auto overhangs = -47.0;
    [[maybe_unused]] constexpr auto overhangs_sq = -9.0;
    [[maybe_unused]] constexpr auto covered_cells = -26.0;
    [[maybe_unused]] constexpr auto covered_cells_sq = 1.0;
    [[maybe_unused]] constexpr auto bumpiness = -7.0;
    [[maybe_unused]] constexpr auto bumpiness_sq = -28.0;
    [[maybe_unused]] constexpr auto height = -46.0;
    [[maybe_unused]] constexpr float well_columns[10] = { 31, 16, -41, 37, 49, 30, 56, 48, -27, 22 };
    [[maybe_unused]] constexpr float clears[5] = { 0, -1700, -100, -50, 490 };
    [[maybe_unused]] constexpr float tspins[4] = { 0, 126, 434, 220 };
    [[maybe_unused]] constexpr float perfect_clear = 200.0;
    [[maybe_unused]] constexpr float wasted_t = -52.0;
    [[maybe_unused]] constexpr float tsd_shape = 180.0;
    [[maybe_unused]] constexpr float well_depth = 91.0; // todo
    [[maybe_unused]] constexpr float max_well_depth = 17.0; // todo
    [[maybe_unused]] constexpr float row_transitions = -5.0;
    [[maybe_unused]] constexpr float donuts = -10.0;
    [[maybe_unused]] constexpr float v_shape = 50.0;
    [[maybe_unused]] constexpr float s_shape = 80.0;
    [[maybe_unused]] constexpr float l_shape = 80.0;
    [[maybe_unused]] constexpr float l2_shape = 80.0;
    [[maybe_unused]] constexpr float counting = 50.0;
}

double Eval::eval_CC(const Board& board, int lines, bool tspin, bool waste_t) {

    using namespace eval_constants;

    double score = 0.0;

    std::pair<int, int> values;

    if (is_top_half(board))
        score += top_half;

    if (is_top_quarter(board))
        score += top_quarter;

    if (is_low(board))
        score += low;

    if (is_perfect_clear(board))
        score += perfect_clear;


    values = cavities_overhangs(board);

    score += values.first * cavity_cells;

    score += values.first * values.first * cavity_cells_sq;

    score += values.second * overhangs;

    score += values.second * values.second * overhangs_sq;

    values = height_features(board);

    score += values.second * height;

    if (ct4(board)) {
        score += counting;
    }

    values = n_covered_cells(board);

    score += values.first * covered_cells;
    score += values.second * covered_cells_sq;

    values = get_bumpiness(board);

    score += values.first * bumpiness;
    score += values.second * bumpiness_sq;

    score += well_columns[well_position(board)];
    score += clears[lines];

    score += get_row_transitions(board) * row_transitions;

    // tspin is set for any piece's spin, so an I piece can spin four lines: the table stops at three
    if (tspin && lines < (int)std::size(tspins)) {
        score += tspins[lines];
    }

    if (waste_t) {
        score += wasted_t;
    }

    return (score + 10000) / 10000;
}

double Eval::eval_CC(Game game, Move move) {
    game.place_piece(move.piece);
    int lines = game.board.clearLines();
    spinType spin = move.piece.spin;
    bool tspin = false;
    if (spin == spinType::normal) {
        tspin = true;
    }

    bool waste_t = false;

    if (move.piece.type == PieceType::T) {
        if (!tspin) {
            waste_t = true;
        }
    }

    return eval_CC(game.board, lines, tspin, waste_t);
}