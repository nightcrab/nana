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
    // eval "convex" only: the whole score of a board with an overhang or cavity above the lowest column
    double nonconvex = -7000;
    // eval "convex" only: prior weight multiplier for actions to such boards, which rank after every convex action
    double nonconvex_prior = 0.01;
    std::array<double, 10> well_columns = { 22.206568640471783, 23.75351666383591, -9.705152993022665,
                                            64.4055163819902,   38.01976098740804, 33.32048980372693,
                                            84.8998702340318,   36.54866592169482, -17.853911656640292,
                                            6.390785928250345 };
    std::array<double, 5> clears = { 0, 329.40179942811164, -2.9680199204562143, -98.8208607848226,
                                     695.6269810587003 };
    std::array<double, 4> tspins = { 0, 80.51857794910987, 159.46297822832895, 325.85611242217686 };

    // Additional nana features start disabled until the unified evaluator is retuned.
    double back_to_back = 0;
    double b2b_clear = 0;
    double combo_garbage = 0;  // actual attack sent, including combo/B2B/PC effects
    double move_time = 0;
    double jeopardy = 0;
    double well_depth = 0;
    int max_well_depth = 17;
    std::array<double, 3> mini_tspins{};  // additive bonuses, as with tspins
    std::array<double, 4> tslot{};

    double c_init = 4.071678062623125;
    double c_base = 7490.956117941063;
    int depth_step = 3;
    double prior_exponent = 0.5152502993309611;
    double n_cap = 2600.4416414699426;
    int depth_cap = 0;
    std::string eval = "nana";
    // false: the search models no opponent, so no speculative garbage, kill reward or belief terms
    bool opponent = false;
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
