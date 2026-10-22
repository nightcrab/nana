#pragma once

#include <array>
#include <string>

// nana's tunable constants: the eval's weights, PUCT and the rollout's score. the defaults are the
// hand-set values; NANA_CONFIG=path.json overrides any of them (read once, on first use). unknown
// keys or malformed values are fatal, so a typo in a tuning run can't silently leave a default in
// place. tuner/ writes these files
struct NanaConfig {
    // eval (Eval::eval_CC): added to the score, which is then mapped by (score + 10000) / 10000
    double top_half = -15.820055334534572;
    double top_quarter = -129.3718779272814;
    double low = -73.32995180401053;
    double cavity_cells = -259.63533157833245;
    double cavity_cells_sq = -16.82791401866494;
    double overhangs = -69.0286103341715;
    double overhangs_sq = -9.351000153849842;
    double covered_cells = -45.3920667793523;
    double covered_cells_sq = -2.1845743836350593;
    double bumpiness = -11.0;
    double bumpiness_sq = -27.688038068690886;
    double height = -58.411659441310874;
    double row_transitions = -10.428933834032492;
    double counting = 64.951011003763;
    double perfect_clear = 211.82984307547167;
    double wasted_t = -26.51295380054306;
    std::array<double, 10> well_columns = { 38.61507034767374, 10.903781585732203, -45.84862477081167,
                                            43.6309383495272,  14.587542614159773, 33.60984910100343,
                                            45.094538189438936, 38.4, -42.89076216657492,
                                            18.33048261479266 };
    std::array<double, 5> clears = { 0, -770.4502697103617, -120.00414559304495, -35.602331509676134,
                                     593.0331936377338 };
    std::array<double, 4> tspins = { 0, 0, 729.3910127543313, 128.41833809365787 };

    // PUCT (UCTNode::c_puct): c = ln((N + c_base + 1) / c_base) + c_init * (1 + depth / depth_step)
    double c_init = 1.25;
    double c_base = 19652;
    int depth_step = 3;
    // priors: the eval's rank r gets weight r^-prior_exponent, normalised
    double prior_exponent = 0.5;
    // an edge's N, R (and M with NANA_FORGET) are scaled by this before each update, so old samples
    // fade geometrically and N is capped at 1 / (1 - decay)
    double decay = 0.9999;

    // rollout score: app_weight * APP + eval_weight * the best eval of the leaf's moves
    double app_weight = 1.0 / 3.0;
    double eval_weight = 0.5;

    // r^-prior_exponent for ranks 1..PRIOR_TABLE (index rank - 1); computed when loaded
    static constexpr int PRIOR_TABLE = 1024;
    std::array<float, PRIOR_TABLE> prior_weight{};
    float priorWeight(int rank) const;

    std::string source = "defaults";
};

// the config in effect: NANA_CONFIG's file, or the defaults
const NanaConfig& config();

// the config as JSON, every key (what --print-config shows)
std::string configJson(const NanaConfig& config);
