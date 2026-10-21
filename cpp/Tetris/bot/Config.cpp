#include "Config.hpp"

#include "Util/json.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

namespace {

[[noreturn]] void fail(const std::string& message) {
    std::cerr << "NANA_CONFIG: " << message << std::endl;
    std::exit(2);
}

// every key, read and written the same way so the two can't drift apart
template <class Visit>
void visitFields(NanaConfig& c, Visit&& visit) {
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
    visit("app_weight", c.app_weight);
    visit("eval_weight", c.eval_weight);
}

template <class T>
void readValue(const nlohmann::json& j, const char* key, T& out) {
    if (!j.is_number()) {
        fail(std::string(key) + " must be a number");
    }
    out = j.get<T>();
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
    if (c.depth_step < 1) {
        fail("depth_step must be at least 1");
    }
    if (!(c.c_base > 0)) {
        fail("c_base must be positive");
    }
    for (int r = 1; r <= NanaConfig::PRIOR_TABLE; r++) {
        c.prior_weight[r - 1] = (float)std::pow((double)r, -c.prior_exponent);
    }
}

NanaConfig load() {
    NanaConfig c;
    const char* path = std::getenv("NANA_CONFIG");
    if (path && *path) {
        std::ifstream in(path);
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
