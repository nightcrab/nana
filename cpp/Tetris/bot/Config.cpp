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
    visit("back_to_back", c.back_to_back);
    visit("b2b_clear", c.b2b_clear);
    visit("combo_garbage", c.combo_garbage);
    visit("move_time", c.move_time);
    visit("jeopardy", c.jeopardy);
    visit("well_depth", c.well_depth);
    visit("max_well_depth", c.max_well_depth);
    visit("mini_tspins", c.mini_tspins);
    visit("tslot", c.tslot);
    visit("c_init", c.c_init);
    visit("c_base", c.c_base);
    visit("depth_step", c.depth_step);
    visit("prior_exponent", c.prior_exponent);
    visit("n_cap", c.n_cap);
    visit("depth_cap", c.depth_cap);
    visit("eval", c.eval);
    visit("opponent", c.opponent);
    visit("neural_model", c.neural_model);
    visit("belief_weights", c.belief_weights);
    visit("opp_cond_u", c.opp_cond_u);
    visit("opp_cond_mean", c.opp_cond_mean);
    visit("opp_cond_v", c.opp_cond_v);
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

// Relative paths are relative to the config file, or to the executable for the defaults.
std::filesystem::path resolvePath(const NanaConfig& c, const std::string& value) {
    std::filesystem::path path = value;
    if (path.is_relative()) {
        path = (c.source == "defaults" ? executableDirectory() : std::filesystem::absolute(c.source).parent_path()) / path;
    }
    return path;
}

void finish(NanaConfig& c) {
    if (c.max_well_depth < 0 || c.max_well_depth > 32) fail("max_well_depth must be in [0, 32]");
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
    if (c.eval != "nana" && c.eval != "convex") {
        fail("eval must be nana or convex; the CC evaluator has been removed");
    }
    if (!(c.nonconvex_prior >= 0 && c.nonconvex_prior <= 1)) {
        fail("nonconvex_prior must be in [0, 1]");
    }
    if (c.depth_cap < 0) {
        fail("depth_cap must be nonnegative");
    }
    if (c.opponent != "none" && c.opponent != "iohmm" && c.opponent != "neural") {
        fail("opponent must be none, iohmm or neural");
    }
    if (!c.neural_model.empty()) {
        c.neural_model = resolvePath(c, c.neural_model).string();
    }
    if (c.opponent != "iohmm") {
        for (double w : c.belief_weights) {
            if (w != 0) fail("belief_weights need the iohmm opponent; set opponent iohmm or zero them");
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
    static const EvalMode mode = config().eval == "convex" ? EvalMode::Convex : EvalMode::Nana;
    return mode;
}

std::string configJson(const NanaConfig& config) {
    NanaConfig c = config;
    nlohmann::ordered_json j;
    visitFields(c, [&](const char* key, auto& field) { j[key] = field; });
    return j.dump();
}
