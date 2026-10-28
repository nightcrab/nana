#pragma once

#include <array>
#include <string>

struct NanaConfig {
    std::string rules = "tetrio";
    int threads = 3;
    std::string pin_layout = "compact";
    std::string variant = "";
    int message_weight = 1;
    int rollout_weight = 30;
    int low_water = 30;
    int give_margin = 3;
    int backoff_iterations = 64;
    int steal_chiplet_penalty = 30;
    int steal_topk = 4;
    int steal_temperature = 10;

    // Unified evaluator release: tuner/releases/unified-cc-20261028.json.
    double top_half = 0.0;
    double top_quarter = -1299.1469468005557;
    double low = -48.85876885951201;
    double cavity_cells = -188.0067334857314;
    double cavity_cells_sq = -11.570201454979951;
    double overhangs = -148.49422381243872;
    double overhangs_sq = -15.506437892380262;
    double covered_cells = -9.488219920162521;
    double covered_cells_sq = -17.734649365123673;
    double bumpiness = -39.19399757617383;
    double bumpiness_sq = -17.479073263689287;
    double height = -74.72915773438541;
    double row_transitions = -24.964292593387626;
    double counting = 123.13225038738523;
    double perfect_clear = 459.1382875053675;
    double wasted_t = -325.3322806159243;
    // eval "convex" only: the whole score of a board with an overhang or cavity above the lowest column
    double nonconvex = -7000;
    // eval "convex" only: prior weight multiplier for actions to such boards, which rank after every convex action
    double nonconvex_prior = 0.01;
    std::array<double, 10> well_columns = {
        -14.027905244456235,
        143.94860745779067,
        105.79624872097673,
        -2.1409300770281163,
        94.62568763831229,
        -13.993603529444357,
        37.377988551410795,
        -13.576156508389971,
        29.175353144995455,
        64.08152580941095,
    };
    std::array<double, 5> clears = {
        0.0,
        -430.9044614913619,
        -82.00526559129185,
        -75.40366226353456,
        568.7142035120609,
    };
    std::array<double, 4> tspins = { 0.0, 211.23662056914796, 449.9985423281276, 805.1118520957763 };

    double back_to_back = 178.00779835117402;
    double b2b_clear = 310.7901540056264;
    double combo_garbage = 163.39949694939244;  // actual attack sent, including combo/B2B/PC effects
    double move_time = -0.9451011861378422;
    double jeopardy = -3.3658024240299156;
    double well_depth = 4.7611948323678055;
    int max_well_depth = 17;
    std::array<double, 3> mini_tspins = { 0.0, 273.6576108015987, 262.13060522082867 };  // additive bonuses, as with tspins
    std::array<double, 4> tslot = { 3.037242442511007, 662.3104264385768, 220.5678195237515, 12.800471695771156 };

    double c_init = 4.071678062623125;
    double c_base = 7490.956117941063;
    int depth_step = 3;
    double prior_exponent = 0.5152502993309611;
    std::string policy = "rank";
    std::string policy_model;
    // Loaded from policy_model, separately from value/reward evaluation weights.
    std::array<double, 43> policy_weights{};
    int policy_max_well_depth = 17;
    double n_cap = 2600.4416414699426;
    int depth_cap = 0;
    std::string eval = "nana";
    // the opponent the search models: none (no speculative garbage, kill reward or belief terms), iohmm or neural
    std::string opponent = "none";
    // the neural opponent's exported model directory
    std::string neural_model;
    std::array<double, 8> belief_weights{};

    double app_weight = 0.6421088128536776;
    double eval_weight = 0.9756232465113018;

    static constexpr int PRIOR_TABLE = 1024;
    std::array<float, PRIOR_TABLE> prior_weight{};
    float priorWeight(int rank) const;

    std::string source = "defaults";
};

// nana_tbp: read nana.json beside the executable (and the TBP-only keys) instead of NANA_CONFIG
void loadTbpConfig();

const NanaConfig& config();

// The unified evaluator, with an optional no-spin style.
enum class EvalMode { Nana, Convex };
EvalMode evalMode();

std::string configJson(const NanaConfig& config);
