#pragma once

#include <array>
#include <string>

// nana's tunable constants: the eval's weights, PUCT and the rollout's score. the defaults are the
// hand-set values; NANA_CONFIG selects weight overrides for tools. TBP uses nana.json beside
// the executable. Settings are read once. Unknown
// keys or malformed values are fatal, so a typo in a tuning run can't silently leave a default in
// place. tuner/ writes these files
// Cold Clear's Standard evaluator's weights (eval "cc", EvalCC.cpp), its defaults for TETR.IO garbage;
// config keys cc_<name>
struct CCWeights {
    double back_to_back = 52;
    double bumpiness = -24;
    double bumpiness_sq = -7;
    double row_transitions = -5;
    double height = -39;
    double top_half = -150;
    double top_quarter = -511;
    double jeopardy = -11;
    double cavity_cells = -173;
    double cavity_cells_sq = -3;
    double overhang_cells = -34;
    double overhang_cells_sq = -1;
    double covered_cells = -17;
    double covered_cells_sq = -1;
    std::array<double, 4> tslot = { 8, 148, 192, 407 };
    double well_depth = 57;
    int max_well_depth = 17;
    std::array<double, 10> well_column = { 20, 23, 20, 50, 59, 21, 59, 10, -10, 24 };

    double move_time = -3;
    double wasted_t = -152;
    double b2b_clear = 104;
    double clear1 = -143;
    double clear2 = -100;
    double clear3 = -58;
    double clear4 = 390;
    double tspin1 = 121;
    double tspin2 = 410;
    double tspin3 = 602;
    double mini_tspin1 = -158;
    double mini_tspin2 = -93;
    double perfect_clear = 999;
    double combo_garbage = 150;
};

struct NanaConfig {
    // Runtime settings for TBP; emulation and benchmarks retain their environment overrides.
    std::string rules = "tetrio";
    int threads = 3;
    std::string reward = "app";
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
    double top_half = -50.809334726527226;
    double top_quarter = -630.2445620700462;
    double low = -33.44361147332671;
    double cavity_cells = -349.58985804116384;
    double cavity_cells_sq = -25.400362612478148;
    double overhangs = -132.6695694448751;
    double overhangs_sq = -1.1524309867331013;
    double covered_cells = -30.498162474880733;
    double covered_cells_sq = -7.8721414742404825;
    double bumpiness = -25.761192630778556;
    double bumpiness_sq = -28.108590764188037;
    double height = -45.15022159907045;
    double row_transitions = -24.575704740846128;
    double counting = 61.9347727280756;
    double perfect_clear = 229.5585551568744;
    double wasted_t = -48.3875036786517;
    std::array<double, 10> well_columns = { 22.206568640471783, 23.75351666383591, -9.705152993022665,
                                            64.4055163819902,   38.01976098740804, 33.32048980372693,
                                            84.8998702340318,   36.54866592169482, -17.853911656640292,
                                            6.390785928250345 };
    std::array<double, 5> clears = { 0, 329.40179942811164, -2.9680199204562143, -98.8208607848226,
                                     695.6269810587003 };
    std::array<double, 4> tspins = { 0, 80.51857794910987, 159.46297822832895, 325.85611242217686 };

    // PUCT (UCTNode::c_puct): c = ln((N + c_base + 1) / c_base) + c_init * (1 + depth / depth_step)
    double c_init = 1.25;
    double c_base = 19652;
    int depth_step = 3;
    // priors: the eval's rank r gets weight r^-prior_exponent, normalised
    double prior_exponent = 0.5;
    // an edge's N and R are scaled by 1 - 1 / n_cap before each update, so old samples fade
    // geometrically (Q averages roughly the last n_cap returns) and N levels off at n_cap
    double n_cap = 10000;
    // nodes this many moves below the root are leaves (0 = no cap)
    int depth_cap = 0;
    // the eval scoring every move: "nana" (eval_CC, the weights above) or "cc" (Cold Clear's Standard
    // evaluator with its default weights, Eval::eval_CC_standard)
    std::string eval = "nana";
    // the CC eval's term over the search's belief about the HMM opponent (EmulationGame::belief): the
    // belief a move leaves, one piece ahead, weighed per hidden state. in CC score units; 0 = off
    std::array<double, 8> belief_weights{};
    CCWeights cc;

    // rollout score: app_weight * APP + eval_weight * the best eval of the leaf's moves
    double app_weight = 0.6421088128536776;
    double eval_weight = 0.9756232465113018;

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
