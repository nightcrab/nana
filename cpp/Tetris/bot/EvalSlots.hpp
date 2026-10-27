#pragma once

#include "engine/Board.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <initializer_list>
#include <optional>

namespace EvalSlots {

// Cold Clear's eval works on 40 rows, so the board's columns are widened to 64 bits (bit y = row y). Cells
// outside the 10 x 40 field count as filled
struct CCBoard {
    static constexpr int HEIGHT = 40;
    static constexpr uint64_t ROWS = (uint64_t(1) << HEIGHT) - 1;
    std::array<uint64_t, 10> cols{};
    std::array<int, 10> heights{};

    static CCBoard from(const Board& board) {
        static_assert(Board::height <= HEIGHT);
        CCBoard b;
        for (int x = 0; x < 10; x++) {
            b.cols[x] = board.board[x];
        }
        b.computeHeights();
        return b;
    }

    void computeHeights() {
        for (int x = 0; x < 10; x++) {
            heights[x] = 64 - std::countl_zero(cols[x]);
        }
    }

    bool occupied(int x, int y) const {
        return x < 0 || y < 0 || x >= 10 || y >= HEIGHT || ((cols[x] >> y) & 1);
    }

    int highest() const {
        return *std::max_element(heights.begin(), heights.end());
    }

    bool surfaceConvex() const {
        // Ignore buried holes below the lowest column, as Board::surface_convex does.
        const int bottom = *std::min_element(heights.begin(), heights.end());
        bool convex = true;
        for (uint64_t col : cols) {
            col >>= bottom;
            convex &= (col & (col + 1)) == 0;
        }
        return convex;
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

inline bool matches(const CCBoard& b, int x, int y, std::initializer_list<const char*> rows) {
    for (const char* row : rows) {
        for (int i = 0; row[i]; i++) {
            if (row[i] == '#' && !b.occupied(x + i, y)) return false;
            if (row[i] == '_' && b.occupied(x + i, y)) return false;
        }
        y--;
    }
    return true;
}

inline std::optional<TSpot> skyTslotRight(const CCBoard& b) {
    for (int x = 0; x + 3 <= 10; x++) {
        const int h1 = b.heights[x + 1], h2 = b.heights[x + 2];
        if (!(h1 <= h2 - 1)) continue;
        if (matches(b, x, h2 + 1, { "#", "_", "#" })) return TSpot{ x + 1, h2, S };
    }
    return std::nullopt;
}

inline std::optional<TSpot> skyTslotLeft(const CCBoard& b) {
    for (int x = 0; x + 3 <= 10; x++) {
        const int h1 = b.heights[x], h2 = b.heights[x + 1];
        if (!(h2 <= h1 - 1)) continue;
        if (matches(b, x, h1 + 1, { "??#", "??_", "??#" })) return TSpot{ x + 1, h1, S };
    }
    return std::nullopt;
}

inline std::optional<TSpot> tstTwistLeft(const CCBoard& b) {
    for (int x = 0; x + 3 <= 10; x++) {
        const int h1 = b.heights[x], h2 = b.heights[x + 1];
        if (!(h1 <= h2 && b.occupied(x - 1, h2) == b.occupied(x - 1, h2 + 1))) continue;
        if (matches(b, x, h2 + 1, { "??#", "??_", "??_", "?__", "??_" })) return TSpot{ x + 2, h2 - 2, W };
    }
    return std::nullopt;
}

inline std::optional<TSpot> tstTwistRight(const CCBoard& b) {
    for (int x = 0; x + 3 <= 10; x++) {
        const int h1 = b.heights[x + 1], h2 = b.heights[x + 2];
        if (!(h2 <= h1 && b.occupied(x + 3, h1) == b.occupied(x + 3, h1 + 1))) continue;
        if (matches(b, x, h1 + 1, { "#", "_", "_", "__", "_" })) return TSpot{ x, h1 - 2, E };
    }
    return std::nullopt;
}

inline std::optional<TSpot> finLeft(const CCBoard& b) {
    for (int x = 0; x + 4 <= 10; x++) {
        const int h1 = b.heights[x], h2 = b.heights[x + 1];
        if (!(h1 <= h2 + 1)) continue;
        if (matches(b, x, h2 + 2, { "??##?", "??__?", "??__#", "??__?", "??#_#" })) return TSpot{ x + 3, h2 - 1, W };
    }
    return std::nullopt;
}

inline std::optional<TSpot> finRight(const CCBoard& b) {
    for (int x = 0; x + 4 <= 10; x++) {
        const int h1 = b.heights[x + 2], h2 = b.heights[x + 3];
        if (!(h2 <= h1 + 1 && b.occupied(x - 1, h1) && b.occupied(x - 1, h1 - 2))) continue;
        if (matches(b, x, h1 + 2, { "##", "__", "__", "__", "_#" })) return TSpot{ x, h1 - 1, E };
    }
    return std::nullopt;
}

inline bool fits(const CCBoard& b, const TSpot& t) {
    for (const auto& c : T_CELLS[t.rot]) {
        if (b.occupied(t.x + c[0], t.y + c[1])) return false;
    }
    return true;
}

inline bool onStack(const CCBoard& b, const TSpot& t) {
    for (const auto& c : T_CELLS[t.rot]) {
        if (b.occupied(t.x + c[0], t.y + c[1] - 1)) return true;
    }
    return false;
}

inline std::optional<TSpot> caveTslot(const CCBoard& b, TSpot start) {
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

inline int cutout(CCBoard& b, const TSpot& t) {
    for (const auto& c : T_CELLS[t.rot]) {
        const int x = t.x + c[0], y = t.y + c[1];
        if (x >= 0 && x < 10 && y >= 0 && y < CCBoard::HEIGHT) {
            b.cols[x] |= uint64_t(1) << y;
        }
    }
    uint64_t full = CCBoard::ROWS;
    for (const uint64_t col : b.cols) {
        full &= col;
    }
    const int lines = std::popcount(full);
    // remove the full rows from the top down, dropping everything above each one
    for (uint64_t rows = full; rows; ) {
        const int y = 63 - std::countl_zero(rows);
        rows &= ~(uint64_t(1) << y);
        const uint64_t below = (uint64_t(1) << y) - 1;
        for (uint64_t& col : b.cols) {
            col = (col & below) | ((col >> 1) & ~below);
        }
    }
    b.computeHeights();
    return lines;
}

inline std::optional<TSpot> findTslot(const CCBoard& b) {
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

