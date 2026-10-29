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

    // CMA-ES evaluator release: tuner/releases/cmaes-cc-20261028.json.
    double top_half = -109.76029518579986;
    double top_quarter = -1803.0356681528551;
    double low = -36.567121160072304;
    double cavity_cells = -147.54340774686182;
    double cavity_cells_sq = -31.90295647410997;
    double overhangs = -16.945937638146518;
    double overhangs_sq = -12.729203193340602;
    double covered_cells = -3.7780475764252373;
    double covered_cells_sq = -12.272199579147944;
    double bumpiness = -2.86597664065836;
    double bumpiness_sq = -2.780480402590669;
    double height = -69.78341839708662;
    double row_transitions = -64.16002378209802;
    double counting = 30.62446007327333;
    double perfect_clear = 439.06511078056377;
    double wasted_t = -317.86291966632245;
    // eval "convex" only: the whole score of a board with an overhang or cavity above the lowest column
    double nonconvex = -7000;
    // eval "convex" only: prior weight multiplier for actions to such boards, which rank after every convex action
    double nonconvex_prior = 0.01;
    std::array<double, 10> well_columns = { -145.68808306207026, 142.67908774675402, 141.40520686624467, 38.072612414358446, -6.682664038102956, -101.41680495789893, -33.88880596406743, -45.24350683293686, -93.46093237745201, 107.8351691546892 };
    std::array<double, 5> clears = { 0.0, -96.98045819408026, -158.93999268422, -205.371315236632, 783.4812191154064 };
    std::array<double, 4> tspins = { 0.0, 330.7953216323606, 409.1553343271181, 439.85461140135305 };

    double back_to_back = 161.82447609775323;
    double b2b_clear = 197.28243633277714;
    double combo_garbage = 249.26862006653437;  // actual attack sent, including combo/B2B/PC effects
    double move_time = -6.217052963848415;
    double jeopardy = -4.756796843837481;
    double well_depth = 70.27551346961596;
    int max_well_depth = 17;
    std::array<double, 3> mini_tspins = { 0.0, 391.548488969957, 229.20103565771637 };  // additive bonuses, as with tspins
    std::array<double, 4> tslot = { 208.54561814201526, 675.5858749391022, 354.6431464881178, 29.080041093900455 };

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

    double app_weight = 0.2559276128014631;
    double eval_weight = 0.36984218083383025;

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
