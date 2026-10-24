#pragma once

#include <array>
#include <string>

// nana's tunable constants: the eval's weights, PUCT and the rollout's score. the defaults are the
// hand-set values; NANA_CONFIG selects weight overrides for tools. TBP uses nana.json beside
// the executable. Settings are read once. Unknown
// keys or malformed values are fatal, so a typo in a tuning run can't silently leave a default in
// place. tuner/ writes these files
struct NanaConfig {
    // Runtime settings for TBP; emulation and benchmarks retain their environment overrides.
    std::string rules = "tetrio";
    int threads = 3;
    bool markov_opponent = true;
    std::string reward = "kill";
    double reward_scale = 0.25;
    double kill_lambda = 0.0;
    double kill_discount = 0.9;
    std::string pin_layout = "compact";
    std::string variant = ""; // empty = choose automatically
    int message_weight = 1;
    int rollout_weight = 30;
    int low_water = 30;
    int give_margin = 3;
    int backoff_iterations = 64;
    int steal_chiplet_penalty = 30;
    int steal_topk = 4;
    int steal_temperature = 10;

    // eval (Eval::eval_CC): added to the score, which is then mapped by (score + 10000) / 10000
    double top_half = -16.5645413218119;
    double top_quarter = -432.6291613236766;
    double low = -37.06215572728501;
    double cavity_cells = -265.79340341552916;
    double cavity_cells_sq = -19.6982738776423;
    double overhangs = -86.02539921248629;
    double overhangs_sq = -2.6534539745444334;
    double covered_cells = -39.17198963527103;
    double covered_cells_sq = -3.2619455153332986;
    double bumpiness = -24.92634243869928;
    double bumpiness_sq = -15.328488400239728;
    double height = -48.080897303757666;
    double row_transitions = -14.964650227838822;
    double counting = 92.92030228516562;
    double perfect_clear = 298.7240888057252;
    double wasted_t = -64.64391867938255;
    std::array<double, 10> well_columns = { 32.670890192719575, 8.052404721512303, -5.624856536146098,
                                            70.57963476394846,  27.060074108181887, 7.992944512649059,
                                            60.30654631101932,  12.59843587982423, -7.048157352276725,
                                            17.660732897688735 };
    std::array<double, 5> clears = { 0, -9.410517988786523, -29.610419738097477, -75.97211687640842,
                                     587.5389313536086 };
    std::array<double, 4> tspins = { 0, 78.50966110273028, 367.1433218304415, 193.4661682079611 };

    // PUCT (UCTNode::c_puct): c = ln((N + c_base + 1) / c_base) + c_init * (1 + depth / depth_step)
    double c_init = 1.25;
    double c_base = 19652;
    int depth_step = 3;
    // priors: the eval's rank r gets weight r^-prior_exponent, normalised
    double prior_exponent = 0.5;
    // an edge's N and R are scaled by 1 - 1 / n_cap before each update, so old samples fade
    // geometrically (Q averages roughly the last n_cap returns) and N levels off at n_cap
    double n_cap = 10000;

    // rollout score: app_weight * APP + eval_weight * the best eval of the leaf's moves
    double app_weight = 1.0 / 3.0;
    double eval_weight = 0.5;

    // r^-prior_exponent for ranks 1..PRIOR_TABLE (index rank - 1); computed when loaded
    static constexpr int PRIOR_TABLE = 1024;
    std::array<float, PRIOR_TABLE> prior_weight{};
    float priorWeight(int rank) const;

    std::string source = "defaults";
};

// Called by the TBP executable/launcher before using settings. Other tools use NANA_CONFIG.
void loadTbpConfig();
// Null for emulation/benchmarks, which retain their environment settings.
const NanaConfig* tbpConfig();

// The configuration in effect.
const NanaConfig& config();

// the config as JSON, every key (what --print-config shows)
std::string configJson(const NanaConfig& config);
