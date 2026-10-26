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

    static CCBoard from(const Board& board) {
        CCBoard b;
        for (int x = 0; x < 10; x++) {
            for (int y = 0; y < (int)Board::height; y++) {
                if (board.get(x, y)) {
                    b.rows[y] |= uint16_t(1u << x);
                }
            }
        }
        b.computeHeights();
        return b;
    }

    void computeHeights() {
        heights.fill(0);
        for (int y = 0; y < 40; y++) {
            for (int x = 0; x < 10; x++) {
                if (rows[y] >> x & 1) {
                    heights[x] = y + 1;
                }
            }
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

double Eval::eval_CC_standard(Game game, Move move, const std::vector<PieceType>& bag_left, double* move_part,
                              BeliefTerm* belief, int meter) {
    static const CCWeights& w = config().cc;
    return eval_CC_standard(w, std::move(game), move, bag_left, move_part, belief, meter);
}

double Eval::eval_CC_standard(const CCWeights& w, Game game, Move move, const std::vector<PieceType>& bag_left,
                              double* move_part, BeliefTerm* belief, [[maybe_unused]] int meter) {
    const Piece& piece = move.piece;
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
        acc += w.perfect_clear;
    }
    else {
        if (did_b2b) acc += w.b2b_clear;
        if (lines > 0) acc += w.combo_garbage * garbage;
        switch (lines) {
        case 1: acc += mini ? w.mini_tspin1 : full ? w.tspin1 : w.clear1; break;
        case 2: acc += mini ? w.mini_tspin2 : full ? w.tspin2 : w.clear2; break;
        case 3: acc += full ? w.tspin3 : w.clear3; break;
        case 4: acc += w.clear4; break;
        default: break;
        }
    }
    if (piece.type == PieceType::T && !(full && lines > 0)) {
        acc += w.wasted_t;
    }
    if (lines > 0) {
        move_time += 40;
    }
    acc += w.move_time * move_time;

    if (game.b2b > 0) {
        transient += w.back_to_back;
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

    CCBoard board = CCBoard::from(game.board);
    int highest = board.highest();
    const double opponent = belief ? belief->value(garbage, lines > 0) : 0.0;
    transient += w.top_quarter * std::max(highest + raised - 15, 0);
    transient += w.top_half * std::max(highest + raised - 10, 0);
    acc += std::trunc(w.jeopardy * std::max(highest + raised - 10, 0) * move_time / 10);

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
        if (cleared < 2) break;
        board = cut;
    }

    highest = board.highest();
    transient += w.height * (highest + raised);

    int well = 0;
    for (int x = 1; x < 10; x++) {
        if (board.heights[x] <= board.heights[well]) well = x;
    }
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
    if (depth != 0) transient += w.well_column[well];

    int transitions = 0;
    for (int y = 0; y < 40; y++) {
        const unsigned r = board.rows[y];
        transitions += std::popcount((r | 0x400u) ^ (1u | (r << 1)));
    }
    transient += w.row_transitions * transitions;

    {
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
    }

    {
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
        transient += w.overhang_cells * overhangs + w.overhang_cells_sq * overhangs * overhangs;
    }

    {
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
    }

    if (move_part) {
        *move_part = acc / 10000.0;
    }
    return (transient + acc + opponent + 10000) / 10000.0;
}

namespace {
int wellColumn(const CCBoard& board) {
    int well = 0;
    for (int x = 1; x < 10; x++) {
        if (board.heights[x] <= board.heights[well]) well = x;
    }
    return well;
}
}

Eval::CCBoardTerms Eval::cc_board_terms(const Board& raw) {
    const CCBoard board = CCBoard::from(raw);
    CCBoardTerms out;
    out.height = board.highest();

    const int well = wellColumn(board);
    for (int y = board.heights[well]; y < 20; y++) {
        bool filled = true;
        for (int x = 0; x < 10 && filled; x++) {
            filled = x == well || board.occupied(x, y);
        }
        if (!filled) break;
        out.well_depth++;
    }

    for (int y = 0; y < 40; y++) {
        const unsigned r = board.rows[y];
        out.row_transitions += std::popcount((r | 0x400u) ^ (1u | (r << 1)));
    }

    int bump = -1;
    int prev = well == 0 ? 1 : 0;
    for (int x = 1; x < 10; x++) {
        if (x == well) continue;
        bump += std::abs(board.heights[prev] - board.heights[x]);
        prev = x;
    }
    out.bumpiness = std::abs(bump);

    for (int y = 0; y < out.height; y++) {
        for (int x = 0; x < 10; x++) {
            if (board.occupied(x, y) || y >= board.heights[x]) continue;
            if (x > 1 && board.heights[x - 1] <= y - 1 && board.heights[x - 2] <= y) { out.overhangs++; continue; }
            if (x < 8 && board.heights[x + 1] <= y - 1 && board.heights[x + 2] <= y) { out.overhangs++; continue; }
            out.cavities++;
        }
    }

    for (int x = 0; x < 10; x++) {
        for (int y = board.heights[x] - 3; y >= 0; y--) {
            if (!board.occupied(x, y)) {
                out.covered += std::min(6, board.heights[x] - y - 1);
            }
        }
    }

    if (const std::optional<TSpot> slot = findTslot(board)) {
        CCBoard cut = board;
        out.tslot = true;
        out.tslot_lines = cutout(cut, *slot);
    }
    return out;
}
