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

// every key, read and written the same way so the two can't drift apart
template <class Visit>
void visitFields(NanaConfig& c, Visit&& visit) {
    if (tbpSettingsEnabled()) {
        visit("rules", c.rules);
        visit("threads", c.threads);
        visit("markov_opponent", c.markov_opponent);
        visit("reward", c.reward);
        visit("reward_scale", c.reward_scale);
        visit("kill_lambda", c.kill_lambda);
        visit("kill_discount", c.kill_discount);
        visit("pin_layout", c.pin_layout);
        visit("variant", c.variant);
        visit("message_weight", c.message_weight);
        visit("rollout_weight", c.rollout_weight);
        visit("low_water", c.low_water);
        visit("give_margin", c.give_margin);
        visit("backoff_iterations", c.backoff_iterations);
        visit("steal_chiplet_penalty", c.steal_chiplet_penalty);
        visit("steal_topk", c.steal_topk);
        visit("steal_temperature", c.steal_temperature);
    }
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
    visit("well_columns", c.well_columns);
    visit("clears", c.clears);
    visit("tspins", c.tspins);
    visit("c_init", c.c_init);
    visit("c_base", c.c_base);
    visit("depth_step", c.depth_step);
    visit("prior_exponent", c.prior_exponent);
    visit("n_cap", c.n_cap);
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
    if (c.reward != "kill" && c.reward != "app") fail("reward must be kill or app");
    if (c.pin_layout != "compact" && c.pin_layout != "spread") fail("pin_layout must be compact or spread");
    if (c.reward_scale <= 0) fail("reward_scale must be positive");
    if (c.kill_lambda < 0 || c.kill_lambda > 1) fail("kill_lambda must be in [0, 1]");
    if (c.kill_discount < 0 || c.kill_discount > 1) fail("kill_discount must be in [0, 1]");
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
        // configs from before n_cap gave its decay, 1 - 1 / n_cap
        if (j.contains("decay") && !j.contains("n_cap")) {
            const double decay = j["decay"].is_number() ? j["decay"].get<double>() : -1;
            if (!(decay >= 0 && decay <= 1)) fail(std::string(path) + ": decay must be in [0, 1]");
            j["n_cap"] = decay < 1 ? 1 / (1 - decay) : std::numeric_limits<double>::max();
            j.erase("decay");
        }
        // every key must be one we know
        size_t known = 0;
        visitFields(c, [&](const char* key, auto& field) {
            if (j.contains(key)) {
                readValue(j[key], key, field);
                known++;
            }
        });
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

} // namespace

void loadTbpConfig() {
    tbpSettingsEnabled() = true;
    (void)config(); // Validate before sending any protocol messages.
}

const NanaConfig* tbpConfig() {
    return tbpSettingsEnabled() ? &config() : nullptr;
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

std::string configJson(const NanaConfig& config) {
    NanaConfig c = config;
    nlohmann::ordered_json j;
    visitFields(c, [&](const char* key, auto& field) { j[key] = field; });
    return j.dump();
}
