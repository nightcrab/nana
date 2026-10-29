#pragma once

#include "OpponentConditioning.hpp"

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

    // CMA-ES evaluator release: tuner/releases/cmaes-nana-20261029.json.
    double top_half = -108.6429596094535;
    double top_quarter = -1405.2750543971713;
    double low = -19.864870582676048;
    double cavity_cells = -91.77055949344265;
    double cavity_cells_sq = -8.53646266169038;
    double overhangs = -60.190125069281635;
    double overhangs_sq = -2.416386445900173;
    double covered_cells = -15.627533669166468;
    double covered_cells_sq = -2.0516440119057293;
    double bumpiness = -9.388409129051226;
    double bumpiness_sq = -1.170462073059369;
    double height = -88.57014967684057;
    double row_transitions = -24.06090474455514;
    double counting = 55.83645713820602;
    double perfect_clear = 1486.9568124836044;
    double wasted_t = -318.97206618852323;
    // eval "convex" only: the whole score of a board with an overhang or cavity above the lowest column
    double nonconvex = -7000;
    // eval "convex" only: prior weight multiplier for actions to such boards, which rank after every convex action
    double nonconvex_prior = 0.01;
    std::array<double, 10> well_columns = { -146.5297662337835, 80.37646306229917, 140.8286535556462, 105.84665865633775, -75.1635200395119, -64.58807277929792, -23.78977077775329, -62.69691476644313, -75.4317857091197, 115.5236771075513 };
    std::array<double, 5> clears = { 0.0, -214.68540714688402, -230.17313232892423, -234.2538780780269, 1195.7104740209197 };
    std::array<double, 4> tspins = { 0.0, 147.8665624737085, 608.4187625948266, 224.7660451061555 };

    double back_to_back = 256.8065601565335;
    double b2b_clear = 327.37576164969494;
    double combo_garbage = 276.5202130368163;  // actual attack sent, including combo/B2B/PC effects
    double move_time = -8.644566216799483;
    double jeopardy = -4.473741555603116;
    double well_depth = 87.65755630388105;
    int max_well_depth = 17;
    std::array<double, 3> mini_tspins = { 0.0, 255.2559154641399, 314.56620282317624 };  // additive bonuses, as with tspins
    std::array<double, 4> tslot = { 328.7276186969281, 728.4895036891403, 627.9114391228678, 214.77109207391828 };

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
    // opponent conditioning (OpponentConditioning.hpp): gate weights u, the opponent features' centre, and the
    // eval direction v
    std::array<double, OpponentFeature::COUNT> opp_cond_u{};
    std::array<double, OpponentFeature::COUNT> opp_cond_mean{};
    std::array<double, PlacementFeature::COUNT> opp_cond_v{};

    double app_weight = 0.4501576877732511;
    double eval_weight = 1.846117229394713;

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
