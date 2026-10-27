#include "Config.hpp"

#include "Util/json.hpp"
#include "Util/ExecutablePath.hpp"
#include <limits>
#include <type_traits>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

namespace {

bool& tbpSettingsEnabled() {
    static bool enabled = false;
    return enabled;
}

[[noreturn]] void fail(const std::string& message) {
    std::cerr << "nana config: " << message << std::endl;
    std::exit(2);
}

template <class Visit>
void visitFields(NanaConfig& c, Visit&& visit) {
    if (tbpSettingsEnabled()) {
        visit("threads", c.threads);
        visit("variant", c.variant);
    }
    visit("rules", c.rules);
    visit("pin_layout", c.pin_layout);
    visit("message_weight", c.message_weight);
    visit("rollout_weight", c.rollout_weight);
    visit("low_water", c.low_water);
    visit("give_margin", c.give_margin);
    visit("backoff_iterations", c.backoff_iterations);
    visit("steal_chiplet_penalty", c.steal_chiplet_penalty);
    visit("steal_topk", c.steal_topk);
    visit("steal_temperature", c.steal_temperature);
    visit("top_half", c.top_half);
    visit("top_quarter", c.top_quarter);
    visit("low", c.low);
    visit("cavity_cells", c.cavity_cells);
    visit("cavity_cells_sq", c.cavity_cells_sq);
    visit("overhangs", c.overhangs);
    visit("overhangs_sq", c.overhangs_sq);
    visit("covered_cells", c.covered_cells);
    visit("covered_cells_sq", c.covered_cells_sq);
    visit("bumpiness", c.bumpiness);
    visit("bumpiness_sq", c.bumpiness_sq);
    visit("height", c.height);
    visit("row_transitions", c.row_transitions);
    visit("counting", c.counting);
    visit("perfect_clear", c.perfect_clear);
    visit("wasted_t", c.wasted_t);
    visit("nonconvex", c.nonconvex);
    visit("nonconvex_prior", c.nonconvex_prior);
    visit("well_columns", c.well_columns);
    visit("clears", c.clears);
    visit("tspins", c.tspins);
    visit("c_init", c.c_init);
    visit("c_base", c.c_base);
    visit("depth_step", c.depth_step);
    visit("prior_exponent", c.prior_exponent);
    visit("n_cap", c.n_cap);
    visit("depth_cap", c.depth_cap);
    visit("eval", c.eval);
    visit("opponent", c.opponent);
    visit("belief_weights", c.belief_weights);
    visit("cc_back_to_back", c.cc.back_to_back);
    visit("cc_bumpiness", c.cc.bumpiness);
    visit("cc_bumpiness_sq", c.cc.bumpiness_sq);
    visit("cc_row_transitions", c.cc.row_transitions);
    visit("cc_height", c.cc.height);
    visit("cc_top_half", c.cc.top_half);
    visit("cc_top_quarter", c.cc.top_quarter);
    visit("cc_jeopardy", c.cc.jeopardy);
    visit("cc_cavity_cells", c.cc.cavity_cells);
    visit("cc_cavity_cells_sq", c.cc.cavity_cells_sq);
    visit("cc_overhang_cells", c.cc.overhang_cells);
    visit("cc_overhang_cells_sq", c.cc.overhang_cells_sq);
    visit("cc_covered_cells", c.cc.covered_cells);
    visit("cc_covered_cells_sq", c.cc.covered_cells_sq);
    visit("cc_tslot", c.cc.tslot);
    visit("cc_well_depth", c.cc.well_depth);
    visit("cc_max_well_depth", c.cc.max_well_depth);
    visit("cc_well_column", c.cc.well_column);
    visit("cc_move_time", c.cc.move_time);
    visit("cc_wasted_t", c.cc.wasted_t);
    visit("cc_b2b_clear", c.cc.b2b_clear);
    visit("cc_clear1", c.cc.clear1);
    visit("cc_clear2", c.cc.clear2);
    visit("cc_clear3", c.cc.clear3);
    visit("cc_clear4", c.cc.clear4);
    visit("cc_tspin1", c.cc.tspin1);
    visit("cc_tspin2", c.cc.tspin2);
    visit("cc_tspin3", c.cc.tspin3);
    visit("cc_mini_tspin1", c.cc.mini_tspin1);
    visit("cc_mini_tspin2", c.cc.mini_tspin2);
    visit("cc_perfect_clear", c.cc.perfect_clear);
    visit("cc_combo_garbage", c.cc.combo_garbage);
    visit("app_weight", c.app_weight);
    visit("eval_weight", c.eval_weight);
}

template <class T>
void readValue(const nlohmann::json& j, const char* key, T& out) {
    if (!j.is_number()) {
        fail(std::string(key) + " must be a number");
    }
    const double value = j.get<double>();
    if (!std::isfinite(value)) fail(std::string(key) + " must be finite");
    if constexpr (std::is_integral_v<T>) {
        if (std::trunc(value) != value || value < std::numeric_limits<T>::lowest() || value > std::numeric_limits<T>::max())
            fail(std::string(key) + " must be an integer in range");
    }
    out = j.get<T>();
}

void readValue(const nlohmann::json& j, const char* key, std::string& out) {
    if (!j.is_string()) fail(std::string(key) + " must be a string");
    out = j.get<std::string>();
}

void readValue(const nlohmann::json& j, const char* key, bool& out) {
    if (!j.is_boolean()) fail(std::string(key) + " must be a boolean");
    out = j.get<bool>();
}

template <class T, size_t N>
void readValue(const nlohmann::json& j, const char* key, std::array<T, N>& out) {
    if (!j.is_array() || j.size() != N) {
        fail(std::string(key) + " must be an array of " + std::to_string(N) + " numbers");
    }
    for (size_t i = 0; i < N; i++) {
        readValue(j[i], key, out[i]);
    }
}

void finish(NanaConfig& c) {
    if (c.rules != "tetrio" && c.rules != "botris") fail("rules must be tetrio or botris");
    if (c.threads < 1) fail("threads must be at least 1");
    if (c.pin_layout != "compact" && c.pin_layout != "spread") fail("pin_layout must be compact or spread");
    if (c.message_weight < 0 || c.rollout_weight < 1 || c.low_water < 0 || c.give_margin < 0 ||
        c.backoff_iterations < 0 || c.steal_chiplet_penalty < 0 || c.steal_temperature < 0)
        fail("request tuning values must be nonnegative, with rollout_weight at least 1");
    if (c.steal_topk < 1 || c.steal_topk > 8) fail("steal_topk must be in [1, 8]");
    if (c.depth_step < 1) {
        fail("depth_step must be at least 1");
    }
    if (!(c.c_base > 0)) {
        fail("c_base must be positive");
    }
    if (!(c.n_cap >= 1)) {
        fail("n_cap must be at least 1");
    }
    if (c.eval != "nana" && c.eval != "cc" && c.eval != "convex") {
        fail("eval must be nana, cc or convex");
    }
    if (!(c.nonconvex_prior >= 0 && c.nonconvex_prior <= 1)) {
        fail("nonconvex_prior must be in [0, 1]");
    }
    if (c.depth_cap < 0) {
        fail("depth_cap must be nonnegative");
    }
    if (!c.opponent) {
        for (double w : c.belief_weights) {
            if (w != 0) fail("belief_weights need an opponent; set opponent true or zero them");
        }
    }
    for (int r = 1; r <= NanaConfig::PRIOR_TABLE; r++) {
        c.prior_weight[r - 1] = (float)std::pow((double)r, -c.prior_exponent);
    }
}


NanaConfig load() {
    NanaConfig c;
    std::filesystem::path file;
    if (tbpSettingsEnabled()) {
        file = executableDirectory() / "nana.json";
        std::error_code ec;
        const bool exists = std::filesystem::exists(file, ec);
        if (ec) fail(file.string() + ": " + ec.message());
        if (!exists) { finish(c); return c; }
    }
    else if (const char* path = std::getenv("NANA_CONFIG"); path && *path) {
        file = path;
    }
    if (!file.empty()) {
        const auto path = file.string();
        std::ifstream in(file);
        if (!in) {
            fail(std::string("can't read ") + path);
        }
        nlohmann::json j;
        try {
            j = nlohmann::json::parse(in);
        }
        catch (const std::exception& e) {
            fail(std::string(path) + ": " + e.what());
        }
        if (!j.is_object()) {
            fail(std::string(path) + " must hold a JSON object");
        }
        size_t known = 0;
        visitFields(c, [&](const char* key, auto& field) {
            if (j.contains(key)) {
                readValue(j[key], key, field);
                known++;
            }
        });
        if (c.eval == "convex" && !j.contains("low_water")) {
            c.low_water = 0;
        }
        if (known != j.size()) {
            std::string unknown;
            for (auto& [key, value] : j.items()) {
                bool found = false;
                visitFields(c, [&](const char* k, auto&) { found |= key == k; });
                if (!found) {
                    unknown += " " + key;
                }
            }
            fail(std::string(path) + ": unknown keys:" + unknown);
        }
        c.source = path;
    }
    finish(c);
    return c;
}

}

void loadTbpConfig() {
    tbpSettingsEnabled() = true;
    (void)config();
}

float NanaConfig::priorWeight(int rank) const {
    if (rank >= 1 && rank <= PRIOR_TABLE) {
        return prior_weight[rank - 1];
    }
    return (float)std::pow((double)rank, -prior_exponent);
}

const NanaConfig& config() {
    static const NanaConfig loaded = load();
    return loaded;
}

EvalMode evalMode() {
    static const EvalMode mode = config().eval == "cc" ? EvalMode::CC
        : config().eval == "convex" ? EvalMode::Convex : EvalMode::Nana;
    return mode;
}

std::string configJson(const NanaConfig& config) {
    NanaConfig c = config;
    nlohmann::ordered_json j;
    visitFields(c, [&](const char* key, auto& field) { j[key] = field; });
    return j.dump();
}
