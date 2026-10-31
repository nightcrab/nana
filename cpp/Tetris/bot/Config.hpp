#pragma once

#include "OpponentConditioning.hpp"

#include <array>
#include <string>
#include <vector>

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

    // SPSA evaluator and conditioning release: tuner/releases/spsa-nana-20261031.json.
    double top_half = -150.77019937724305;
    double top_quarter = -1447.5344860029204;
    double low = -19.319518883088975;
    double cavity_cells = -82.82988185903105;
    double cavity_cells_sq = -0.3722831516732441;
    double overhangs = 0.0;
    double overhangs_sq = -7.9020054335493555;
    double covered_cells = -26.015844396455876;
    double covered_cells_sq = 0.0;
    double bumpiness = -16.018266360514602;
    double bumpiness_sq = 0.0;
    double height = -75.44623816056239;
    double row_transitions = -21.334725296646553;
    double counting = 32.25935456575195;
    double perfect_clear = 1817.413696501668;
    double wasted_t = -265.7204705129369;
    // eval "convex" only: the whole score of a board with an overhang or cavity above the lowest column
    double nonconvex = -7000;
    // eval "convex" only: prior weight multiplier for actions to such boards, which rank after every convex action
    double nonconvex_prior = 0.01;
    std::array<double, 10> well_columns = { -118.30840844589497, 78.37369483554878, 150.0, 143.3345685051998, -37.662806580672026, -38.64459987201468, 2.691431537453809, -59.69983015195314, -55.934001417849814, 99.44029023975449 };
    std::array<double, 5> clears = { 0.0, -308.9073238367056, -208.99939164640375, -202.97621229207584, 1295.4992715154945 };
    std::array<double, 4> tspins = { 0.0, 204.9211049448751, 404.2997816931169, 285.87872249360703 };

    double back_to_back = 199.07698809776554;
    double b2b_clear = 288.0919658165247;
    double combo_garbage = 259.1726824231862;  // actual attack sent, including combo/B2B/PC effects
    double move_time = -9.15348607705981;
    double jeopardy = -10.09845812787632;
    double well_depth = 88.96981893076922;
    int max_well_depth = 17;
    std::array<double, 3> mini_tspins = { 0.0, 367.5442059739438, 359.52396170489624 };  // additive bonuses, as with tspins
    std::array<double, 4> tslot = { 369.3438912520636, 754.168976494279, 638.1968038206053, 209.0889847807848 };
    // the T-slot score is multiplied by max(0, 1 + tslot_t_distance * d), d = pieces placed before a T can be
    double tslot_t_distance = 0;
    // per cavity cell in the top half (row 10 and up), on top of cavity_cells
    double high_cavity_cells = 0;
    // per row the tallest of columns 3-6 stands above 12
    double spawn_height = 0;
    // an I or T in hold
    double hold_i = 0;
    double hold_t = 0;

    double c_init = 4.071678062623125;
    double c_base = 7490.956117941063;
    int depth_step = 3;
    // Leaf-evaluation rank prior: weight(rank) = rank ** -prior_exponent.
    double prior_exponent = 0.5152502993309611;
    double n_cap = 2600.4416414699426;
    int depth_cap = 0;
    std::string eval = "nana";
    // the opponent the search models: none (no speculative garbage, kill reward or belief terms), iohmm or neural
    std::string opponent = "none";
    // the neural opponent's exported model directory
    std::string neural_model;
    std::array<double, 8> belief_weights{};
    // opponent conditioning (OpponentConditioning.hpp): one row per rank-1 term of gate weights u, the opponent
    // features' centre, the eval direction v and the reward weight shift. SPSA release: tuner/releases/spsa-nana-20261031.json.
    std::vector<std::array<double, OpponentFeature::COUNT>> opp_cond_u = {{
        0.0263265204844757, 0.06528980408421325, 0.048540678876267396, -0.07619859482736954,
        -0.0015338227813016768, -0.0010015167652430812, 0.012947094025306444, 0.0012131839745147966,
        0.0021494668150928847, 0.14760291230569872, -0.020593028849696215, -0.23806680089567778,
        -0.13991839218391566, -0.6675366692606723, -0.6865137243722634, -0.02851227603488224,
        -0.08164093657332738, -0.10861775506284799, 0.022123220907802388
    }, {}};
    std::vector<std::array<double, OpponentFeature::COUNT>> opp_cond_mean = {{
        7.547427128721338, 2.3591821554132433, 11.348346419273177, 8.528617687041397,
        23.389072583880278, 260.61884180832965, 23.75563002309904, 87.60023173933278,
        75.77549961539673, 4.033677849387274, 10.799402744645247, 0.7339754229099711,
        0.3222937049191404, 0.420630495371318, 0.6752409274604635, 0.6536904094879747,
        13.87685052701389, 9.541085842450318, 2.6921151869340516
    }, {
        7.547427128721338, 2.3591821554132433, 11.348346419273177, 8.528617687041397,
        23.389072583880278, 260.61884180832965, 23.75563002309904, 87.60023173933278,
        75.77549961539673, 4.033677849387274, 10.799402744645247, 0.7339754229099711,
        0.3222937049191404, 0.420630495371318, 0.6752409274604635, 0.6536904094879747,
        13.87685052701389, 9.541085842450318, 2.6921151869340516
    }};
    std::vector<std::array<double, PlacementFeature::COUNT>> opp_cond_v = {{
        -254.39845554119745, 557.1718874857977, -6.424720087371867, -33.19684344383819,
        -1016.4041321377433, 171.7415292500168, 21.358682147520824, -485.1653574512817,
        672.2245526462143, 168.4009878427056, -64.09988273016825, -64.08647420240106,
        -194.50300586926642, -34.51538381236338, 72.4221144866168, 35.7524451411187,
        6.180811355730597, 3.4081165231025246, -392.72159757393774, 114.25605570175594,
        -16.439735912561705, -154.99759989117214, -7.080995158835364, 64.99150348905249,
        -2.8062680315418387, -70.3020575043075, 4.936790186009944, -29.869151658785082,
        2.666545379156717, 2.521796029154828, -22.183382327041723, -103.95498969166013,
        -43.572685564788415, -76.39659650863604, -7.418439218619426, 12.828304697192578,
        -98.67831303288645, 24.502210307052252, -10.263801211223637, -63.44391019770761,
        12.46418617412428, -84.36262999407339, 29.469326635552108, -171.7584970465165,
        223.05803616239007, 124.14214184578668, -211.3763913501424, 159.16343006322745
    }, {}};
    // each term's shift of [app_weight, eval_weight]; rows missing from the end are zero
    std::vector<std::array<double, 2>> opp_cond_reward;

    double app_weight = 0.5050599468172452;
    double eval_weight = 1.1190433022782837;

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
