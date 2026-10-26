#include "Eval.hpp"
#include "Config.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <optional>

namespace {

struct CCBoard {
    std::array<uint16_t, 40> rows{};
    std::array<int, 10> heights{};

    // heights from each column's top bit, rows by transposing the columns up to the tallest one
    static CCBoard from(const Board& board) {
        static_assert(Board::height <= 40);
        CCBoard b;
        int top = 0;
        for (int x = 0; x < 10; x++) {
            b.heights[x] = (int)Board::height - std::countl_zero(board.board[x]);
            top = std::max(top, b.heights[x]);
        }
        for (int y = 0; y < top; y++) {
            unsigned row = 0;
            for (int x = 0; x < 10; x++) {
                row |= ((board.board[x] >> y) & 1u) << x;
            }
            b.rows[y] = uint16_t(row);
        }
        return b;
    }

    // scans down from the top row, setting each column's height at its first filled cell
    void computeHeights() {
        heights.fill(0);
        unsigned found = 0;
        for (int y = 39; y >= 0 && found != 0x3FFu; y--) {
            for (unsigned fresh = rows[y] & ~found; fresh; fresh &= fresh - 1) {
                heights[std::countr_zero(fresh)] = y + 1;
            }
            found |= rows[y];
        }
    }

    bool occupied(int x, int y) const {
        return x < 0 || y < 0 || x >= 10 || y >= 40 || (rows[y] >> x & 1);
    }

    int highest() const {
        return *std::max_element(heights.begin(), heights.end());
    }
};

enum CCRot { N = 0, E = 1, S = 2, W = 3 };
constexpr int T_CELLS[4][4][2] = {
    { { -1, 0 }, { 0, 0 }, { 1, 0 }, { 0, 1 } },
    { { 0, 1 }, { 0, 0 }, { 0, -1 }, { 1, 0 } },
    { { 1, 0 }, { 0, 0 }, { -1, 0 }, { 0, -1 } },
    { { 0, -1 }, { 0, 0 }, { 0, 1 }, { -1, 0 } },
};

struct TSpot {
    int x, y, rot;
};

bool matches(const CCBoard& b, int x, int y, std::initializer_list<const char*> rows) {
    for (const char* row : rows) {
        for (int i = 0; row[i]; i++) {
            if (row[i] == '#' && !b.occupied(x + i, y)) return false;
            if (row[i] == '_' && b.occupied(x + i, y)) return false;
        }
        y--;
    }
    return true;
}

std::optional<TSpot> skyTslotRight(const CCBoard& b) {
    for (int x = 0; x + 3 <= 10; x++) {
        const int h1 = b.heights[x + 1], h2 = b.heights[x + 2];
        if (!(h1 <= h2 - 1)) continue;
        if (matches(b, x, h2 + 1, { "#", "_", "#" })) return TSpot{ x + 1, h2, S };
    }
    return std::nullopt;
}

std::optional<TSpot> skyTslotLeft(const CCBoard& b) {
    for (int x = 0; x + 3 <= 10; x++) {
        const int h1 = b.heights[x], h2 = b.heights[x + 1];
        if (!(h2 <= h1 - 1)) continue;
        if (matches(b, x, h1 + 1, { "??#", "??_", "??#" })) return TSpot{ x + 1, h1, S };
    }
    return std::nullopt;
}

std::optional<TSpot> tstTwistLeft(const CCBoard& b) {
    for (int x = 0; x + 3 <= 10; x++) {
        const int h1 = b.heights[x], h2 = b.heights[x + 1];
        if (!(h1 <= h2 && b.occupied(x - 1, h2) == b.occupied(x - 1, h2 + 1))) continue;
        if (matches(b, x, h2 + 1, { "??#", "??_", "??_", "?__", "??_" })) return TSpot{ x + 2, h2 - 2, W };
    }
    return std::nullopt;
}

std::optional<TSpot> tstTwistRight(const CCBoard& b) {
    for (int x = 0; x + 3 <= 10; x++) {
        const int h1 = b.heights[x + 1], h2 = b.heights[x + 2];
        if (!(h2 <= h1 && b.occupied(x + 3, h1) == b.occupied(x + 3, h1 + 1))) continue;
        if (matches(b, x, h1 + 1, { "#", "_", "_", "__", "_" })) return TSpot{ x, h1 - 2, E };
    }
    return std::nullopt;
}

std::optional<TSpot> finLeft(const CCBoard& b) {
    for (int x = 0; x + 4 <= 10; x++) {
        const int h1 = b.heights[x], h2 = b.heights[x + 1];
        if (!(h1 <= h2 + 1)) continue;
        if (matches(b, x, h2 + 2, { "??##?", "??__?", "??__#", "??__?", "??#_#" })) return TSpot{ x + 3, h2 - 1, W };
    }
    return std::nullopt;
}

std::optional<TSpot> finRight(const CCBoard& b) {
    for (int x = 0; x + 4 <= 10; x++) {
        const int h1 = b.heights[x + 2], h2 = b.heights[x + 3];
        if (!(h2 <= h1 + 1 && b.occupied(x - 1, h1) && b.occupied(x - 1, h1 - 2))) continue;
        if (matches(b, x, h1 + 2, { "##", "__", "__", "__", "_#" })) return TSpot{ x, h1 - 1, E };
    }
    return std::nullopt;
}

bool fits(const CCBoard& b, const TSpot& t) {
    for (const auto& c : T_CELLS[t.rot]) {
        if (b.occupied(t.x + c[0], t.y + c[1])) return false;
    }
    return true;
}

bool onStack(const CCBoard& b, const TSpot& t) {
    for (const auto& c : T_CELLS[t.rot]) {
        if (b.occupied(t.x + c[0], t.y + c[1] - 1)) return true;
    }
    return false;
}

std::optional<TSpot> caveTslot(const CCBoard& b, TSpot start) {
    while (true) {
        TSpot down = start;
        down.y--;
        if (!fits(b, down)) break;
        start = down;
    }
    const int x = start.x, y = start.y;
    if (start.rot == E) {
        if (!b.occupied(x - 1, y) && b.occupied(x - 1, y - 1) && b.occupied(x + 1, y - 1) && b.occupied(x - 1, y + 1)) {
            return TSpot{ x, y, S };
        }
        if (!b.occupied(x + 1, y - 1) && !b.occupied(x + 2, y - 1) && !b.occupied(x + 1, y - 2) && b.occupied(x - 1, y)
            && b.occupied(x + 2, y) && b.occupied(x, y - 2) && b.occupied(x + 2, y - 2)) {
            return TSpot{ x + 1, y - 1, S };
        }
        return std::nullopt;
    }
    if (start.rot == W) {
        if (!b.occupied(x + 1, y) && b.occupied(x + 1, y + 1) && b.occupied(x + 1, y - 1) && b.occupied(x - 1, y - 1)) {
            return TSpot{ x, y, S };
        }
        if (!b.occupied(x - 1, y - 1) && !b.occupied(x - 2, y - 1) && !b.occupied(x - 1, y - 2) && b.occupied(x + 1, y)
            && b.occupied(x - 2, y) && b.occupied(x - 2, y - 2) && b.occupied(x, y - 2)) {
            return TSpot{ x - 1, y - 1, S };
        }
    }
    return std::nullopt;
}

int cutout(CCBoard& b, const TSpot& t) {
    for (const auto& c : T_CELLS[t.rot]) {
        const int x = t.x + c[0], y = t.y + c[1];
        if (x >= 0 && x < 10 && y >= 0 && y < 40) {
            b.rows[y] |= uint16_t(1u << x);
        }
    }
    int lines = 0;
    int to = 0;
    for (int y = 0; y < 40; y++) {
        if (b.rows[y] == 0x3FF) {
            lines++;
            continue;
        }
        b.rows[to++] = b.rows[y];
    }
    while (to < 40) {
        b.rows[to++] = 0;
    }
    b.computeHeights();
    return lines;
}

std::optional<TSpot> findTslot(const CCBoard& b) {
    if (auto t = skyTslotLeft(b)) return t;
    if (auto t = skyTslotRight(b)) return t;
    std::optional<TSpot> tst = tstTwistLeft(b);
    if (!tst) tst = tstTwistRight(b);
    if (tst) {
        if (auto t = caveTslot(b, *tst)) return t;
        const int corners = b.occupied(tst->x - 1, tst->y - 1) + b.occupied(tst->x + 1, tst->y - 1)
                          + b.occupied(tst->x - 1, tst->y + 1) + b.occupied(tst->x + 1, tst->y + 1);
        if (corners >= 3 && onStack(b, *tst)) return tst;
    }
    if (auto t = finLeft(b)) return t;
    return finRight(b);
}

}

Eval::BeliefTerm::BeliefTerm(const OpponentFilter* filter, int meter) : filter(filter), meter(meter) {
    cache.fill(std::numeric_limits<float>::quiet_NaN());
}

float Eval::BeliefTerm::value(int attack, bool cleared) {
    float& v = cache[(cleared ? 16 : 0) + std::min(attack, 15)];
    if (std::isnan(v)) {
        v = filter->valueAfter(attack, cleared, cleared ? 0 : meter, config().belief_weights);
    }
    return v;
}

#ifndef NANA_EVAL_TANK
#define NANA_EVAL_TANK 0
#endif

namespace {

enum Term {
    PerfectClear, B2bClear, ComboGarbage, Clear, SpinClear, WastedT, MoveTime, Jeopardy, BackToBack, TopQuarter,
    TopHalf, Tslot, Height, WellDepth, WellColumn, RowTransitions, Bumpiness, BumpinessSq, CavityCells, CavityCellsSq,
    OverhangCells, OverhangCellsSq, CoveredCells, CoveredCellsSq, TermCount
};
static_assert(TermCount == Eval::CC_TERMS);

constexpr uint32_t bit(Term t) { return 1u << t; }
constexpr uint32_t AllTerms = (1u << TermCount) - 1;
// terms that need the CCBoard
constexpr uint32_t BoardTerms = AllTerms & ~(bit(PerfectClear) | bit(B2bClear) | bit(ComboGarbage) | bit(Clear)
    | bit(SpinClear) | bit(WastedT) | bit(MoveTime) | bit(BackToBack));

// Record: also add each weighted term into terms. Mask: the terms computed, the rest are skipped
template <bool Record, uint32_t Mask = AllTerms>
double ccScore(Game game, const Piece& piece, const std::vector<PieceType>& bag_left, double* move_part,
               Eval::BeliefTerm* belief, [[maybe_unused]] int meter, [[maybe_unused]] double* terms) {
    static const CCWeights& w = config().cc;
    constexpr auto has = [](uint32_t terms_wanted) { return (Mask & terms_wanted) != 0; };
    auto rec = [&]([[maybe_unused]] Term t, [[maybe_unused]] double v) {
        if constexpr (Record) terms[t] += v;
    };
    const auto b2b_before = game.b2b;

    const bool held = piece.type != game.current_piece.type;
    static constexpr int rotations[4] = { 0, 1, 2, 1 };
    int move_time = std::abs(piece.position.x - (10 / 2 - 1)) + rotations[(int)piece.rotation] + (held ? 1 : 0);

    game.place_piece(piece);
    const int lines = game.board.clearLines();
    spinType spin = piece.spin;
    if (std::holds_alternative<TetrioS1>(game.mode) && piece.type != PieceType::T) {
        spin = spinType::null;
    }
    bool pc = true;
    for (int x = 0; x < Board::width; x++) {
        pc = pc && game.board.get_column(x) == 0;
    }
    const int garbage = game.damage_sent(lines, spin, pc);

    const bool full = spin == spinType::normal;
    const bool mini = spin == spinType::mini;
    const bool hard = lines == 4 || spin != spinType::null;
    const bool did_b2b = lines > 0 && hard && b2b_before > 0;

    double acc = 0;
    double transient = 0;

    if (pc) {
        if constexpr (has(bit(PerfectClear))) {
            acc += w.perfect_clear;
            rec(PerfectClear, w.perfect_clear);
        }
    }
    else {
        if constexpr (has(bit(B2bClear))) {
            if (did_b2b) {
                acc += w.b2b_clear;
                rec(B2bClear, w.b2b_clear);
            }
        }
        if constexpr (has(bit(ComboGarbage))) {
            if (lines > 0) {
                acc += w.combo_garbage * garbage;
                rec(ComboGarbage, w.combo_garbage * garbage);
            }
        }
        if constexpr (has(bit(Clear) | bit(SpinClear))) {
            double clear = 0;
            switch (lines) {
            case 1: clear = mini ? w.mini_tspin1 : full ? w.tspin1 : w.clear1; break;
            case 2: clear = mini ? w.mini_tspin2 : full ? w.tspin2 : w.clear2; break;
            case 3: clear = full ? w.tspin3 : w.clear3; break;
            case 4: clear = w.clear4; break;
            default: break;
            }
            acc += clear;
            rec((mini || full) && lines < 4 ? SpinClear : Clear, clear);
        }
    }
    if constexpr (has(bit(WastedT))) {
        if (piece.type == PieceType::T && !(full && lines > 0)) {
            acc += w.wasted_t;
            rec(WastedT, w.wasted_t);
        }
    }
    if (lines > 0) {
        move_time += 40;
    }
    if constexpr (has(bit(MoveTime))) {
        acc += w.move_time * move_time;
        rec(MoveTime, w.move_time * move_time);
    }

    if constexpr (has(bit(BackToBack))) {
        if (game.b2b > 0) {
            transient += w.back_to_back;
            rec(BackToBack, w.back_to_back);
        }
    }

    const int tank = NANA_EVAL_TANK && lines == 0 && meter > 0
        ? (std::holds_alternative<TetrioS1>(game.mode) ? meter : std::min(meter, 8)) : 0;
    if constexpr (NANA_EVAL_TANK == 2) {
        if (tank > 0) {
            uint32_t h = 0;
            for (int x = 0; x < Board::width; x++) {
                h = h * 31 + game.board.get_column(x);
            }
            game.add_garbage(tank, (int)(h % Board::width));
        }
    }
    const int raised = NANA_EVAL_TANK == 1 ? tank : 0;

    const double opponent = belief ? belief->value(garbage, lines > 0) : 0.0;
    if constexpr (has(BoardTerms)) {
    CCBoard board = CCBoard::from(game.board);
    int highest = board.highest();
    if constexpr (has(bit(TopQuarter))) {
        transient += w.top_quarter * std::max(highest + raised - 15, 0);
        rec(TopQuarter, w.top_quarter * std::max(highest + raised - 15, 0));
    }
    if constexpr (has(bit(TopHalf))) {
        transient += w.top_half * std::max(highest + raised - 10, 0);
        rec(TopHalf, w.top_half * std::max(highest + raised - 10, 0));
    }
    if constexpr (has(bit(Jeopardy))) {
        acc += std::trunc(w.jeopardy * std::max(highest + raised - 10, 0) * move_time / 10);
        rec(Jeopardy, std::trunc(w.jeopardy * std::max(highest + raised - 10, 0) * move_time / 10));
    }

    if constexpr (has(bit(Tslot))) {
    unsigned bag = 0;
    for (PieceType p : bag_left) {
        bag |= 1u << (int)p;
    }
    auto refill = [&](PieceType p) {
        if (p == PieceType::Empty) return;
        if (bag == 0x7F) bag = 0;
        bag |= 1u << (int)p;
    };
    for (auto it = game.queue.rbegin(); it != game.queue.rend(); ++it) {
        refill(*it);
    }
    refill(game.current_piece.type);
    const int ts = (int)((bag >> (int)PieceType::T) & 1) + (std::popcount(bag) <= 3 ? 1 : 0)
                 + (game.hold == PieceType::T ? 1 : 0);

    for (int i = 0; i < ts; i++) {
        const std::optional<TSpot> slot = findTslot(board);
        if (!slot) break;
        CCBoard cut = board;
        const int cleared = cutout(cut, *slot);
        transient += w.tslot[std::min(cleared, 3)];
        rec(Tslot, w.tslot[std::min(cleared, 3)]);
        if (cleared < 2) break;
        board = cut;
    }
    }

    highest = board.highest();
    if constexpr (has(bit(Height))) {
        transient += w.height * (highest + raised);
        rec(Height, w.height * (highest + raised));
    }

    int well = 0;
    if constexpr (has(bit(WellDepth) | bit(WellColumn) | bit(Bumpiness) | bit(BumpinessSq))) {
        for (int x = 1; x < 10; x++) {
            if (board.heights[x] <= board.heights[well]) well = x;
        }
    }
    if constexpr (has(bit(WellDepth) | bit(WellColumn))) {
    int depth = 0;
    for (int y = board.heights[well]; y < 20; y++) {
        bool filled = true;
        for (int x = 0; x < 10 && filled; x++) {
            filled = x == well || board.occupied(x, y);
        }
        if (!filled) break;
        depth++;
    }
    depth = std::min(depth, w.max_well_depth);
    transient += w.well_depth * depth;
    rec(WellDepth, w.well_depth * depth);
    if (depth != 0) {
        transient += w.well_column[well];
        rec(WellColumn, w.well_column[well]);
    }
    }

    if constexpr (has(bit(RowTransitions))) {
    int transitions = 0;
    for (int y = 0; y < 40; y++) {
        const unsigned r = board.rows[y];
        transitions += std::popcount((r | 0x400u) ^ (1u | (r << 1)));
    }
    transient += w.row_transitions * transitions;
    rec(RowTransitions, w.row_transitions * transitions);
    }

    if constexpr (has(bit(Bumpiness) | bit(BumpinessSq))) {
        int bump = -1, bump_sq = -1;
        int prev = well == 0 ? 1 : 0;
        for (int x = 1; x < 10; x++) {
            if (x == well) continue;
            const int dh = std::abs(board.heights[prev] - board.heights[x]);
            bump += dh;
            bump_sq += dh * dh;
            prev = x;
        }
        transient += std::abs(bump) * w.bumpiness + std::abs(bump_sq) * w.bumpiness_sq;
        rec(Bumpiness, std::abs(bump) * w.bumpiness);
        rec(BumpinessSq, std::abs(bump_sq) * w.bumpiness_sq);
    }

    if constexpr (has(bit(CavityCells) | bit(CavityCellsSq) | bit(OverhangCells) | bit(OverhangCellsSq))) {
        int cavities = 0, overhangs = 0;
        for (int y = 0; y < highest; y++) {
            for (int x = 0; x < 10; x++) {
                if (board.occupied(x, y) || y >= board.heights[x]) continue;
                if (x > 1 && board.heights[x - 1] <= y - 1 && board.heights[x - 2] <= y) { overhangs++; continue; }
                if (x < 8 && board.heights[x + 1] <= y - 1 && board.heights[x + 2] <= y) { overhangs++; continue; }
                cavities++;
            }
        }
        transient += w.cavity_cells * cavities + w.cavity_cells_sq * cavities * cavities;
        rec(CavityCells, w.cavity_cells * cavities);
        rec(CavityCellsSq, w.cavity_cells_sq * cavities * cavities);
        transient += w.overhang_cells * overhangs + w.overhang_cells_sq * overhangs * overhangs;
        rec(OverhangCells, w.overhang_cells * overhangs);
        rec(OverhangCellsSq, w.overhang_cells_sq * overhangs * overhangs);
    }

    if constexpr (has(bit(CoveredCells) | bit(CoveredCellsSq))) {
        int covered = 0, covered_sq = 0;
        for (int x = 0; x < 10; x++) {
            for (int y = board.heights[x] - 3; y >= 0; y--) {
                if (!board.occupied(x, y)) {
                    const int cells = std::min(6, board.heights[x] - y - 1);
                    covered += cells;
                    covered_sq += cells * cells;
                }
            }
        }
        transient += w.covered_cells * covered + w.covered_cells_sq * covered_sq;
        rec(CoveredCells, w.covered_cells * covered);
        rec(CoveredCellsSq, w.covered_cells_sq * covered_sq);
    }
    }

    if (move_part) {
        *move_part = acc / 10000.0;
    }
    return (transient + acc + opponent + 10000) / 10000.0;
}

constexpr uint32_t Squares = bit(BumpinessSq) | bit(CavityCellsSq) | bit(OverhangCellsSq) | bit(CoveredCellsSq);
constexpr uint32_t NearZero = bit(PerfectClear) | bit(B2bClear) | bit(BackToBack) | bit(Jeopardy) | bit(WellColumn)
    | bit(SpinClear);
constexpr uint32_t Recommended = bit(CavityCells) | bit(CoveredCells) | bit(TopQuarter) | bit(TopHalf) | bit(Height)
    | bit(BumpinessSq) | bit(RowTransitions) | bit(WellDepth) | bit(Clear) | bit(ComboGarbage) | bit(MoveTime)
    | bit(WastedT) | bit(Tslot);
constexpr uint32_t Compact = bit(CavityCells) | bit(Height) | bit(TopHalf) | bit(TopQuarter) | bit(Bumpiness)
    | bit(RowTransitions) | bit(WellDepth) | bit(WastedT) | bit(Tslot) | bit(Clear) | bit(ComboGarbage) | bit(MoveTime);

struct Timed {
    const char* name;
    uint32_t mask;
};
constexpr Timed timed_sets[] = {
    { "full", AllTerms },
    { "no squares", AllTerms & ~Squares },
    { "no squares, one hole term", AllTerms & ~Squares & ~bit(CoveredCells) },
    { "no squares, one hole term, no near-zero", AllTerms & ~Squares & ~bit(CoveredCells) & ~NearZero },
    { "recommended 13", Recommended },
    { "recommended 13 without spin terms", Recommended & ~(bit(WastedT) | bit(Tslot)) },
    { "compact 12", Compact },
    { "compact without spin terms", Compact & ~(bit(WastedT) | bit(Tslot)) },
    { "base only (no terms)", 0 },
};
constexpr int timed_set_count = (int)std::size(timed_sets);

using TimedFn = double (*)(Game, const Piece&, const std::vector<PieceType>&, double*, Eval::BeliefTerm*, int, double*);

template <size_t... I>
constexpr std::array<TimedFn, sizeof...(I)> setTable(std::index_sequence<I...>) {
    return { &ccScore<false, timed_sets[I].mask>... };
}
template <size_t... I>
constexpr std::array<TimedFn, sizeof...(I)> singleTable(std::index_sequence<I...>) {
    return { &ccScore<false, (1u << I)>... };
}
constexpr auto set_fns = setTable(std::make_index_sequence<timed_set_count>{});
constexpr auto single_fns = singleTable(std::make_index_sequence<TermCount>{});

}

int Eval::ccTimedCount() {
    return timed_set_count + TermCount;
}

int Eval::ccTimedSets() {
    return timed_set_count;
}

const char* Eval::ccTimedName(int i) {
    if (i < timed_set_count) return timed_sets[i].name;
    return ccTermNames[i - timed_set_count];
}

double Eval::evalCCTimed(int i, const Game& game, const Piece& piece, const std::vector<PieceType>& bag_left) {
    const TimedFn fn = i < timed_set_count ? set_fns[i] : single_fns[i - timed_set_count];
    return fn(game, piece, bag_left, nullptr, nullptr, 0, nullptr);
}

double Eval::evalCC(const Game& game, const Piece& piece, const std::vector<PieceType>& bag_left, double* move_part,
                    BeliefTerm* belief, int meter) {
    return ccScore<false>(game, piece, bag_left, move_part, belief, meter, nullptr);
}

double Eval::evalCCTerms(const Game& game, const Piece& piece, const std::vector<PieceType>& bag_left,
                         std::array<double, CC_TERMS>& terms) {
    terms.fill(0);
    return ccScore<true>(game, piece, bag_left, nullptr, nullptr, 0, terms.data());
}
