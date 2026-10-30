#include "Eval.hpp"
#include "Config.hpp"
#include "EvalSlots.hpp"
#include <cmath>
#include <limits>
#include <tuple>

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
static bool quadReady(const Board& board, int bottom, int top) {
    if (top != bottom + 4) return false;
    int wells = 0;
    for (uint32_t col : board.board) {
        const uint64_t above = uint64_t(col) >> bottom;
        if (above == 0) ++wells;
        else if (above != 0b1111) return false;
    }
    return wells == 1;
}

bool Eval::ct4(const Board& board) {
    const auto [bottom, top] = height_features(board);
    return quadReady(board, bottom, top);
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

Eval::BoardFeatures Eval::boardFeatures(const Board& board) {
    BoardFeatures f{};
    std::tie(f.bottom, f.height) = height_features(board);
    std::tie(f.cavities, f.overhangs) = cavities_overhangs(board);
    std::tie(f.covered, f.covered_sq) = n_covered_cells(board);
    std::tie(f.bumpiness, f.bumpiness_sq) = get_bumpiness(board);
    f.transitions = get_row_transitions(board);
    f.well = well_position(board);
    f.well_depth = well_depth(board, f.well, 32);
    f.top_half = is_top_half(board);
    f.top_quarter = is_top_quarter(board);
    f.low = is_low(board);
    f.perfect_clear = is_perfect_clear(board);
    f.quad_ready = quadReady(board, f.bottom, f.height);
    return f;
}

Eval::MoveFeatures Eval::moveFeatures(const Game& parent, const Piece& piece, Game& after, int lines) {
    MoveFeatures f{};
    f.lines = lines;
    f.spin = std::holds_alternative<TetrioS1>(after.mode) && piece.type != PieceType::T
        ? spinType::null : piece.spin;
    f.perfect_clear = after.board.is_empty();
    f.attack = after.damage_sent(lines, f.spin, f.perfect_clear);
    f.waste_t = piece.type == PieceType::T && f.spin != spinType::normal;
    f.back_to_back = after.b2b > 0;
    f.b2b_clear = lines > 0 && (lines == 4 || f.spin != spinType::null) && parent.b2b > 0;
    static constexpr int rotations[] = {0, 1, 2, 1};
    f.time = std::abs(piece.position.x - 4) + rotations[int(piece.rotation)]
           + (piece.type != parent.current_piece.type) + (lines > 0 ? 40 : 0);
    return f;
}

// Apply value weights in the original order to preserve scoring arithmetic.
template <bool Convex>
static double evalBoard(const Eval::BoardFeatures& f, int lines, bool tspin, bool waste_t, const NanaConfig& c) {
    double score = 0.0;
    if (f.top_half) score += c.top_half;
    if (f.top_quarter) score += c.top_quarter;
    if (f.low) score += c.low;
    if (f.perfect_clear) score += c.perfect_clear;
    score += f.cavities * c.cavity_cells;
    score += f.cavities * f.cavities * c.cavity_cells_sq;
    if constexpr (!Convex) {
        score += f.overhangs * c.overhangs;
        score += f.overhangs * f.overhangs * c.overhangs_sq;
    }
    score += f.height * c.height;
    if (f.quad_ready) score += c.counting;
    score += f.covered * c.covered_cells;
    score += f.covered_sq * c.covered_cells_sq;
    score += f.bumpiness * c.bumpiness;
    score += f.bumpiness_sq * c.bumpiness_sq;
    score += c.well_columns[f.well];
    score += c.well_depth * std::min(f.well_depth, c.max_well_depth);
    score += c.clears[lines];
    score += f.transitions * c.row_transitions;
    if (!Convex && tspin && lines < (int)std::size(c.tspins)) score += c.tspins[lines];
    if (!Convex && waste_t) score += c.wasted_t;
    return (score + 10000) / 10000;
}

double Eval::nanaBoard(const Board& board, int lines, bool tspin, bool waste_t) {
    return evalBoard<false>(boardFeatures(board), lines, tspin, waste_t, config());
}

namespace {

struct Slots {
    std::array<int, 3> clears{};
    int count = 0;
    double score(const std::array<double, 4>& weights) const {
        double result = 0;
        for (int i = 0; i < count; ++i) result += weights[clears[i]];
        return result;
    }
};

Slots slotFeatures(const Game& after, const std::vector<PieceType>& bag_left) {
    auto board = EvalSlots::CCBoard::from(after.board);
    if (board.surfaceConvex()) return {};
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
    Slots slots;
    for (int i = 0; i < available; ++i) {
        if (board.surfaceConvex()) break;
        const auto slot = EvalSlots::findTslot(board);
        if (!slot) break;
        const int cleared = EvalSlots::cutout(board, *slot);
        slots.clears[slots.count++] = std::min(cleared, 3);
        if (cleared < 2) break;
    }
    // Only slot potential uses the hypothetical cutouts; all other features use the actual board.
    return slots;
}

}

namespace {

bool nonzero(const auto& weights) {
    return std::any_of(weights.begin(), weights.end(), [](double w) { return w != 0; });
}

// Some row has both a nonzero u and a nonzero v.
bool conditioningActive() {
    static const bool active = [] {
        const auto& c = config();
        for (size_t k = 0; k < c.opp_cond_v.size(); ++k)
            if (nonzero(c.opp_cond_u[k]) && nonzero(c.opp_cond_v[k])) return true;
        return false;
    }();
    return active;
}

// Whether Eval::opponent_direction is nonzero, and whether it weights the T-slot features.
bool direction_active = false;
bool direction_uses_slots = false;

struct Conditioned {
    double move = 0, board = 0;
};

// w · f for one placement, split into the move's share and the board's share.
Conditioned conditionedFeatures(const Eval::BoardFeatures& f, const Eval::MoveFeatures& m, const Slots& slots,
                                bool no_spin, const NanaConfig& c, const Eval::ConditioningDirection& w) {
    namespace P = PlacementFeature;
    std::array<double, P::COUNT> x{};
    const bool full = m.spin == spinType::normal, mini = m.spin == spinType::mini;
    x[P::CLEARS + m.lines] = 1;
    if (!no_spin && full && m.lines < 4) x[P::TSPINS + m.lines] = 1;
    if (!no_spin && mini && m.lines < 3) x[P::MINI_TSPINS + m.lines] = 1;
    x[P::PERFECT_CLEAR] = m.perfect_clear;
    x[P::WASTED_T] = !no_spin && m.waste_t;
    x[P::B2B_CLEAR] = m.b2b_clear;
    x[P::ATTACK] = m.lines > 0 ? m.attack : 0;
    x[P::TIME] = m.time;
    x[P::JEOPARDY] = std::max(f.height - 10, 0) * m.time / 10.0;
    x[P::TOP_HALF] = f.top_half;
    x[P::TOP_QUARTER] = f.top_quarter;
    x[P::LOW] = f.low;
    x[P::CAVITIES] = f.cavities;
    x[P::CAVITIES_SQ] = f.cavities * f.cavities;
    if (!no_spin) {
        x[P::OVERHANGS] = f.overhangs;
        x[P::OVERHANGS_SQ] = f.overhangs * f.overhangs;
    }
    x[P::HEIGHT] = f.height;
    x[P::QUAD_READY] = f.quad_ready;
    x[P::COVERED] = f.covered;
    x[P::COVERED_SQ] = f.covered_sq;
    x[P::BUMPINESS] = f.bumpiness;
    x[P::BUMPINESS_SQ] = f.bumpiness_sq;
    x[P::WELL_COLUMNS + f.well] = 1;
    x[P::WELL_DEPTH] = std::min(f.well_depth, c.max_well_depth);
    x[P::TRANSITIONS] = f.transitions;
    x[P::BACK_TO_BACK] = m.back_to_back;
    if (!no_spin) {
        for (int i = 0; i < slots.count; ++i) x[P::TSLOT + slots.clears[i]] += 1;
    }
    Conditioned out;
    for (int i = 0; i < P::BOARD; ++i) out.move += w[i] * x[i];
    for (int i = P::BOARD; i < P::COUNT; ++i) out.board += w[i] * x[i];
    return out;
}

double scorePlacement(const Eval::BoardFeatures& features, const Eval::MoveFeatures& move_features,
                      const Slots& slots, double belief, bool no_spin, const NanaConfig& c, double* move_part) {
    const int lines=move_features.lines, attack=move_features.attack;
    const bool full=move_features.spin==spinType::normal, mini=move_features.spin==spinType::mini;
    const bool pc=move_features.perfect_clear, waste_t=move_features.waste_t;
    constexpr double alive=1.0/10000;
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

    if (move_features.b2b_clear) {
        extra_move += c.b2b_clear;
    }

    if (lines > 0) {
        extra_move += c.combo_garbage * attack;
    }

    const int time = move_features.time;

    extra_move += c.move_time * time;
    extra_move += std::trunc(c.jeopardy * std::max(features.height - 10, 0) * time / 10);

    double position = move_features.back_to_back ? c.back_to_back : 0;

    if (!no_spin) position += slots.score(c.tslot);

    position += belief;

    if (direction_active) {
        const Conditioned conditioned =
            conditionedFeatures(features, move_features, slots, no_spin, c, Eval::opponent_direction);
        extra_move += conditioned.move;
        position += conditioned.board;
    }

    if (move_part) {
        *move_part = (move + extra_move) / 10000;
    }

    const double board = no_spin
        ? evalBoard<true>(features, lines, false, false, c)
        : evalBoard<false>(features, lines, full, waste_t, c);

    const double total = board + (extra_move + position) / 10000;
    return no_spin ? std::max(total, alive) : total;
}

} // namespace

double Eval::evalNana(const Context& context, const Piece& piece, double* move_part, bool* convex) {
    const auto& c = config();
    const bool no_spin = context.mode == EvalMode::Convex;

    Game after = context.game;
    after.place_piece(piece);
    const int lines = after.board.clearLines();
    const auto move_features = moveFeatures(context.game, piece, after, lines);
    const int attack = move_features.attack;

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

    const auto features = boardFeatures(after.board);
    Slots slots;
    if (!no_spin && (nonzero(c.tslot) || direction_uses_slots))
        slots = slotFeatures(after, context.bag_left);
    const double belief = context.belief ? context.belief->value(attack, lines > 0) : 0;
    return scorePlacement(features, move_features, slots, belief, no_spin, c, move_part);
}

std::vector<float> Eval::opponentGates(const Board& board, int meter, int b2b, int combo) {
    namespace O = OpponentFeature;
    const BoardFeatures f = boardFeatures(board);
    std::array<double, O::COUNT> o{};
    o[O::BOTTOM] = f.bottom;
    o[O::HEIGHT] = f.height;
    o[O::CAVITIES] = f.cavities;
    o[O::OVERHANGS] = f.overhangs;
    o[O::COVERED] = f.covered;
    o[O::COVERED_SQ] = f.covered_sq;
    o[O::BUMPINESS] = f.bumpiness;
    o[O::BUMPINESS_SQ] = f.bumpiness_sq;
    o[O::TRANSITIONS] = f.transitions;
    o[O::WELL] = f.well;
    o[O::WELL_DEPTH] = f.well_depth;
    o[O::TOP_HALF] = f.top_half;
    o[O::TOP_QUARTER] = f.top_quarter;
    o[O::LOW] = f.low;
    o[O::PERFECT_CLEAR] = f.perfect_clear;
    o[O::QUAD_READY] = f.quad_ready;
    o[O::METER] = meter;
    o[O::B2B] = b2b;
    o[O::COMBO] = combo;
    const auto& c = config();
    std::vector<float> gates(c.opp_cond_v.size());
    for (size_t k = 0; k < gates.size(); ++k) {
        double g = 0;
        for (int i = 0; i < O::COUNT; ++i) g += c.opp_cond_u[k][i] * (o[i] - c.opp_cond_mean[k][i]);
        gates[k] = (float)std::clamp(g, -1.0, 1.0);
    }
    return gates;
}

Eval::ConditioningDirection Eval::opponentDirection(const Board& board, int meter, int b2b, int combo) {
    ConditioningDirection w{};
    if (!conditioningActive()) return w;
    const auto& v = config().opp_cond_v;
    const std::vector<float> gates = opponentGates(board, meter, b2b, combo);
    for (size_t k = 0; k < gates.size(); ++k)
        for (int i = 0; i < PlacementFeature::COUNT; ++i) w[i] += (double)gates[k] * v[k][i];
    return w;
}

void Eval::setOpponentDirection(const ConditioningDirection& direction) {
    if (direction != opponent_direction) {
        opponent_direction = direction;
        direction_active = nonzero(direction);
        direction_uses_slots = std::any_of(direction.begin() + PlacementFeature::TSLOT, direction.end(),
                                           [](double w) { return w != 0; });
        ++q_epoch;
    }
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

Eval::ActionEvaluation Eval::evaluateAction(const Context& context, const Piece& piece) {
    double move = 0;
    bool convex = true;
    const float value = (float)evalNana(context, piece, &move, &convex);
    return {value, (float)move, convex};
}
