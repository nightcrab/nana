#include "Eval.hpp"
#include "Config.hpp"
#include "EvalSlots.hpp"
#include <cmath>
#include <limits>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "engine/Board.hpp"
#include "engine/Game.hpp"
#include "Move.hpp"

Eval::BeliefTerm::BeliefTerm(const opponent::Belief* filter, int meter) : filter(filter), meter(meter) {
    cache.fill(std::numeric_limits<float>::quiet_NaN());
}

float Eval::BeliefTerm::value(int attack, bool cleared) {
    // Large attacks must not share the cached value for attack 15.
    if (attack > 15) return filter->value_after(attack, cleared, cleared ? 0 : meter, config().belief_weights);
    float& v = cache[(cleared ? 16 : 0) + attack];
    if (std::isnan(v)) {
        v = filter->value_after(attack, cleared, cleared ? 0 : meter, config().belief_weights);
    }
    return v;
}

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

// All buried empty cells are cavities. The overhang feature additionally counts
// those with an open side at the same row, treating both directions equally.
std::pair<int, int> Eval::cavities_overhangs(const Board& board) {
    int cavities = 0, overhangs = 0;
    for (int x = 0; x < Board::width; ++x) {
        const uint64_t col = board.board[x];
        const int height = 32 - std::countl_zero(board.board[x]);
        const uint64_t holes = ~col & ((uint64_t(1) << height) - 1);
        uint64_t open_side = 0;
        if (x > 0) open_side |= ~uint64_t(0) << (32 - std::countl_zero(board.board[x - 1]));
        if (x + 1 < Board::width) open_side |= ~uint64_t(0) << (32 - std::countl_zero(board.board[x + 1]));
        cavities += std::popcount(holes);
        overhangs += std::popcount(holes & open_side);
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
    int covered = 0, covered_sq = 0;
    for (uint32_t col : board.board) {
        const int height = 32 - std::countl_zero(col);
        uint64_t holes = ~uint64_t(col) & ((uint64_t(1) << height) - 1);
        for (; holes; holes &= holes - 1) {
            const int y = std::countr_zero(holes);
            const int cells = std::popcount(uint64_t(col) >> (y + 1));
            covered += cells;
            covered_sq += cells * cells;
        }
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

// Four complete rows beside a single well above the common stack base.
bool Eval::ct4(const Board& board) {
    const auto [bottom, top] = height_features(board);
    if (top != bottom + 4) return false;
    int wells = 0;
    for (uint32_t col : board.board) {
        const uint64_t above = uint64_t(col) >> bottom;
        if (above == 0) ++wells;
        else if (above != 0b1111) return false;
    }
    return wells == 1;
}

int Eval::get_row_transitions(const Board& board) {
    int transitions = std::popcount(~board.board.front()) + std::popcount(~board.board.back());
    for (int x = 1; x < Board::width; ++x) {
        transitions += std::popcount(board.board[x - 1] ^ board.board[x]);
    }
    return transitions;
}

int Eval::well_depth(const Board& board, int well, int cap) {
    uint32_t beside = UINT32_MAX;
    for (int x = 0; x < Board::width; ++x) {
        if (x != well) beside &= board.board[x];
    }
    const int bottom = 32 - std::countl_zero(board.board[well]);
    // Preserve CC's visible-field limit for the new well-depth feature.
    if (bottom >= 20) return 0;
    return std::min({std::countr_one(beside >> bottom), 20 - bottom, cap});
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

    const int well = Eval::well_position(board);
    score += c.well_columns[well];
    score += c.well_depth * Eval::well_depth(board, well, c.max_well_depth);
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

namespace {

double slotPotential(const Game& after, const std::vector<PieceType>& bag_left) {
    const auto& weights = config().tslot;
    if (std::all_of(weights.begin(), weights.end(), [](double w) { return w == 0; })) return 0;
    auto board = EvalSlots::CCBoard::from(after.board);
    if (board.surfaceConvex()) return 0;
    unsigned bag = 0;
    for (PieceType p : bag_left) {
        if (p != PieceType::Empty) bag |= 1u << (int)p;
    }
    auto refill = [&](PieceType p) {
        if (p == PieceType::Empty) return;
        if (bag == 0x7F) bag = 0;
        bag |= 1u << (int)p;
    };
    for (auto it = after.queue.rbegin(); it != after.queue.rend(); ++it) refill(*it);
    refill(after.current_piece.type);
    const int available = ((bag >> (int)PieceType::T) & 1) + (std::popcount(bag) <= 3 ? 1 : 0)
                        + (after.hold == PieceType::T ? 1 : 0);
    double score = 0;
    for (int i = 0; i < available; ++i) {
        if (board.surfaceConvex()) break;
        const auto slot = EvalSlots::findTslot(board);
        if (!slot) break;
        const int cleared = EvalSlots::cutout(board, *slot);
        score += weights[std::min(cleared, 3)];
        if (cleared < 2) break;
    }
    // Only slot potential uses the hypothetical cutouts; all other features use the actual board.
    return score;
}

}

double Eval::evalNana(const Context& context, const Piece& piece, double* move_part, bool* convex) {
    const auto& c = config();
    const bool no_spin = context.mode == EvalMode::Convex;

    Game after = context.game;
    const int b2b_before = after.b2b;
    const bool held = piece.type != after.current_piece.type;

    after.place_piece(piece);
    const int lines = after.board.clearLines();

    spinType spin = piece.spin;
    if (std::holds_alternative<TetrioS1>(after.mode) && piece.type != PieceType::T) {
        spin = spinType::null;
    }

    const bool full = spin == spinType::normal;
    const bool mini = spin == spinType::mini;
    const bool pc = after.board.is_empty();
    const int attack = after.damage_sent(lines, spin, pc);
    const bool waste_t = piece.type == PieceType::T && !full;

    const auto surface = EvalSlots::CCBoard::from(after.board);
    const bool is_convex = surface.surfaceConvex();

    if (convex) {
        *convex = no_spin ? is_convex : true;
    }

    constexpr double alive = 1.0 / 10000;
    if (no_spin && !is_convex) {
        if (move_part) {
            *move_part = 0;
        }

        return std::max(1 + c.nonconvex / 10000, alive);
    }

    // nanaBoard includes the legacy move features; expose the same amounts for search backup.
    double move = c.clears[lines] + (pc ? c.perfect_clear : 0);

    if (!no_spin && full && lines < (int)c.tspins.size()) {
        move += c.tspins[lines];
    }

    if (!no_spin && waste_t) {
        move += c.wasted_t;
    }

    double extra_move = 0;

    if (!no_spin && mini && lines < (int)c.mini_tspins.size()) {
        extra_move += c.mini_tspins[lines];
    }

    if (lines > 0 && (lines == 4 || spin != spinType::null) && b2b_before > 0) {
        extra_move += c.b2b_clear;
    }

    if (lines > 0) {
        extra_move += c.combo_garbage * attack;
    }

    static constexpr int rotations[] = { 0, 1, 2, 1 };
    const int time = std::abs(piece.position.x - 4)
                   + rotations[(int)piece.rotation]
                   + held
                   + (lines > 0 ? 40 : 0);

    extra_move += c.move_time * time;
    extra_move += std::trunc(c.jeopardy * std::max(surface.highest() - 10, 0) * time / 10);

    double position = after.b2b > 0 ? c.back_to_back : 0;

    if (!no_spin) {
        position += slotPotential(after, context.bag_left);
    }

    if (context.belief) {
        position += context.belief->value(attack, lines > 0);
    }

    if (move_part) {
        *move_part = (move + extra_move) / 10000;
    }

    const double board = no_spin
        ? evalBoard<true>(after.board, lines, false, false)
        : nanaBoard(after.board, lines, full, waste_t);

    const double total = board + (extra_move + position) / 10000;
    return no_spin ? std::max(total, alive) : total;
}

double Eval::evalNana(const Game& game, const Piece& piece, double* move_part) {
    return evalNana(Context{game, EvalMode::Nana}, piece, move_part);
}

double Eval::evalConvex(const Game& game, const Piece& piece, double* move_part, bool* convex) {
    return evalNana(Context{game, EvalMode::Convex}, piece, move_part, convex);
}

double Eval::evaluate(const Context& context, const Piece& piece, double* move_part, bool* convex) {
    return evalNana(context, piece, move_part, convex);
}
