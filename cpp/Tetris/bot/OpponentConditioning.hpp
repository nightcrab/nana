#pragma once

// Opponent conditioning: at the root, g = clamp(u · (o − mean), -1, 1) from the opponent's visible state o, fixed
// for the search; every placement's eval moves by g × (v · f). Config opp_cond_u and opp_cond_mean are indexed by
// OpponentFeature, opp_cond_v by PlacementFeature. All weights zero (the default) turns the term off.

// The opponent's board features, as Eval::boardFeatures computes them, then its TBP2 meter, B2B and combo.
namespace OpponentFeature {
enum : int {
    BOTTOM,         // 0
    HEIGHT,         // 1
    CAVITIES,       // 2
    OVERHANGS,      // 3
    COVERED,        // 4
    COVERED_SQ,     // 5
    BUMPINESS,      // 6
    BUMPINESS_SQ,   // 7
    TRANSITIONS,    // 8
    WELL,           // 9   well column index
    WELL_DEPTH,     // 10
    TOP_HALF,       // 11
    TOP_QUARTER,    // 12
    LOW,            // 13
    PERFECT_CLEAR,  // 14
    QUAD_READY,     // 15
    METER,          // 16  garbage queued against the opponent
    B2B,            // 17
    COMBO,          // 18
    COUNT           // 19
};
}

// One placement's features, mirroring the eval's linear terms. The move's own features come first and count
// toward move_part; the board's start at BOARD. Spin features are zero in the convex eval, as its weights are.
namespace PlacementFeature {
enum : int {
    CLEARS = 0,                     // 0-4    lines cleared is 0..4 (one-hot, like `clears`)
    TSPINS = CLEARS + 5,            // 5-8    full T-spin clearing 0..3 lines
    MINI_TSPINS = TSPINS + 4,       // 9-11   mini T-spin clearing 0..2 lines
    PERFECT_CLEAR = MINI_TSPINS + 3,// 12
    WASTED_T,                       // 13
    B2B_CLEAR,                      // 14
    ATTACK,                         // 15     attack sent by a clearing move
    TIME,                           // 16
    JEOPARDY,                       // 17     max(height - 10, 0) * time / 10
    BOARD,                          // 18     first board feature
    TOP_HALF = BOARD,               // 18
    TOP_QUARTER,                    // 19
    LOW,                            // 20
    CAVITIES,                       // 21
    CAVITIES_SQ,                    // 22
    OVERHANGS,                      // 23
    OVERHANGS_SQ,                   // 24
    HEIGHT,                         // 25
    QUAD_READY,                     // 26
    COVERED,                        // 27
    COVERED_SQ,                     // 28
    BUMPINESS,                      // 29
    BUMPINESS_SQ,                   // 30
    WELL_COLUMNS,                   // 31-40  well column 0..9 (one-hot)
    WELL_DEPTH = WELL_COLUMNS + 10, // 41     capped at max_well_depth
    TRANSITIONS,                    // 42
    BACK_TO_BACK,                   // 43
    TSLOT,                          // 44-47  T-slots clearing 0..3 lines (counts)
    COUNT = TSLOT + 4               // 48
};
}
