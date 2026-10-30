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
    // features' centre, the eval direction v and the reward weight shift. CMA-ES release: tuner/releases/cmaes-conditioning-20261030.json.
    std::vector<std::array<double, OpponentFeature::COUNT>> opp_cond_u = {{
        0.03542803559339122, 0.04577614885916531, 0.028558418802087843, -0.050848202358357235,
        -0.0017657944389820462, -0.00043110851925596194, 0.015215895054372014, 0.001265879314512719,
        0.0018871814540236155, 0.16831169868740792, -0.016833785541009227, 0.1142518808174553,
        -0.049626090794986144, -0.5447605958640336, -0.9477324332906636, 0.044489177263846935,
        -0.09907023107216564, -0.129768723804215, 0.03248317650248825
    }, {}};
    std::vector<std::array<double, OpponentFeature::COUNT>> opp_cond_mean = {{
        7.521459631982796, 0.023327230551095646, 11.330412430495574, 7.6358984337979,
        22.896463165066745, 235.4890508300948, 28.379833991813282, 85.96490686465252,
        69.22950285966837, 2.6847793117988363, 12.10348132772627, 0.8457113307737033,
        0.45113318496516897, 0.5507021250542614, 0.6644215089860201, 0.5546224699751728,
        14.432517388650668, 8.168744166988391, 4.276417810573616
    }, {
        7.521459631982796, 0.023327230551095646, 11.330412430495574, 7.6358984337979,
        22.896463165066745, 235.4890508300948, 28.379833991813282, 85.96490686465252,
        69.22950285966837, 2.6847793117988363, 12.10348132772627, 0.8457113307737033,
        0.45113318496516897, 0.5507021250542614, 0.6644215089860201, 0.5546224699751728,
        14.432517388650668, 8.168744166988391, 4.276417810573616
    }};
    std::vector<std::array<double, PlacementFeature::COUNT>> opp_cond_v = {{
        -169.04441838267007, 467.1718874857977, -40.338485206425844, -58.91212674807599,
        -731.0669377897611, 72.5290077460972, 40.73669057874605, -855.0874226825116,
        710.4088289993224, 13.821471017514, -17.104291357577722, -61.932978187776655,
        -151.9694499699558, -168.91284386349153, -35.208602077502974, 83.8550641138612,
        4.1330431415080895, 3.1059208778829657, -349.90528576631084, 440.5080721293093,
        -6.250516389870569, -138.68937516944447, -13.772742237433832, 116.94602708065065,
        -3.464891151998529, -35.69912571831418, 5.695240595125483, -19.58796180535097,
        2.6567918195142113, -4.393274169819975, -28.681948827972036, -143.81100661552583,
        -96.72936983340364, -90.06179167051859, 14.98660112599947, -29.210935095014534,
        -75.47621179522174, 11.643785343012865, 9.643343135764724, -59.663414625494084,
        -27.41336546707427, -62.03559427852019, 17.76592219673482, -108.6363168667132,
        177.3981870312531, -52.3311648290711, -231.73569496286962, 151.3361819532863
    }, {}};
    // each term's shift of [app_weight, eval_weight]; rows missing from the end are zero
    std::vector<std::array<double, 2>> opp_cond_reward;

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
