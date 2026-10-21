#pragma once

#include <array>
#include <string>

// nana's tunable constants: the eval's weights, PUCT and the rollout's score. the defaults are the
// hand-set values; NANA_CONFIG=path.json overrides any of them (read once, on first use). unknown
// keys or malformed values are fatal, so a typo in a tuning run can't silently leave a default in
// place. tuner/ writes these files
struct NanaConfig {
    // eval (Eval::eval_CC): added to the score, which is then mapped by (score + 10000) / 10000
    double top_half = -130.0;
    double top_quarter = -499.0;
    double low = -50.0;
    double cavity_cells = -176.0;
    double cavity_cells_sq = -6.0;
    double overhangs = -47.0;
    double overhangs_sq = -9.0;
    double covered_cells = -26.0;
    double covered_cells_sq = 1.0;
    double bumpiness = -7.0;
    double bumpiness_sq = -28.0;
    double height = -46.0;
    double row_transitions = -5.0;
    double counting = 50.0;
    double perfect_clear = 200.0;
    double wasted_t = -52.0;
    std::array<double, 10> well_columns = { 31, 16, -41, 37, 49, 30, 56, 48, -27, 22 };
    std::array<double, 5> clears = { 0, -1700, -100, -50, 490 };
    std::array<double, 4> tspins = { 0, 126, 434, 220 };

    // PUCT (UCTNode::c_puct): c = ln((N + c_base + 1) / c_base) + c_init * (1 + depth / depth_step)
    double c_init = 1.25;
    double c_base = 19652;
    int depth_step = 3;
    // priors: the eval's rank r gets weight r^-prior_exponent, normalised
    double prior_exponent = 0.5;

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
