// Runs nana's CC search (NANA_CC_SEARCH) on one position and dumps the whole tree, every selection
// step and the position with the future the search played, for scripts/cc_tree_compare.py to set
// against Cold Clear's tree on the same position.
// usage: nana_cctree [--seed S] [--pieces P] [--position FILE] [--sims N] [--search-seed X]
//                    [--lockstep-seed L] [--future K] [--canonical] [--clear-meter]
// The position is game S after P greedy moves under the CC eval, or the first line of FILE in
// nana_eval_dump's format (board, queue [current, next...], hold, bag left after the queue, b2b,
// combo). --canonical ranks ties by occupied cells and weighs in integer score units, as
// tree_dump --canonical does, so the two stay in step past arbitrary tie orders. --clear-meter
// drops the position's incoming garbage, which Cold Clear's search can't see. Needs NANA_CONFIG
// with "eval": "cc". Writes one JSON object to stdout.

#include "CCTrace.hpp"
#include "Config.hpp"
#include "EmulateCommon.hpp"
#include "EmulationGame.hpp"
#include "Eval.hpp"
#include "Search.hpp"
#include "UCT.hpp"
#include "Util/FastSeed.hpp"
#include "Util/json.hpp"
#include "engine/Utility.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

static_assert(cc_search && NANA_CC_LOCKSTEP, "build nana_cctree with NANA_CC_SEARCH=1 and NANA_CC_LOCKSTEP=1");

#ifndef NANA_LOAD_FACTOR
#define NANA_LOAD_FACTOR 6
#endif

static const char* pieceName(PieceType type) {
    switch (type) {
    case PieceType::S: return "S";
    case PieceType::Z: return "Z";
    case PieceType::J: return "J";
    case PieceType::L: return "L";
    case PieceType::T: return "T";
    case PieceType::O: return "O";
    case PieceType::I: return "I";
    default: return "?";
    }
}

static PieceType pieceType(const std::string& name) {
    for (PieceType t : { PieceType::S, PieceType::Z, PieceType::J, PieceType::L, PieceType::T, PieceType::O, PieceType::I }) {
        if (name == pieceName(t)) return t;
    }
    std::cerr << "unknown piece " << name << std::endl;
    std::exit(2);
}

static const char* rotationName(RotationDirection r) {
    switch (r) {
    case RotationDirection::North: return "north";
    case RotationDirection::East: return "east";
    case RotationDirection::South: return "south";
    default: return "west";
    }
}

// a placement as both tools name it: type, orientation, x, y (TBP coordinates), spin (T only)
static std::string moveKey(const Piece& piece) {
    const char* spin = "none";
    if (piece.type == PieceType::T) {
        spin = piece.spin == spinType::normal ? "full" : piece.spin == spinType::mini ? "mini" : "none";
    }
    return std::string(pieceName(piece.type)) + " " + rotationName(piece.rotation) + " " + std::to_string(piece.position.x) +
           " " + std::to_string(piece.position.y) + " " + spin;
}

static std::string hex(uint32_t h) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%08x", h);
    return buf;
}

static nlohmann::json ccUnits(float R) {
    if (R <= CC_DEAD) return nullptr;
    return std::lround((R - 1.0) * 10000.0);
}

// the cells a placement fills, in TBP coordinates (libtetris PieceState::cells), sorted
static std::array<std::pair<int, int>, 4> cells(const Piece& piece) {
    static constexpr int north[7][4][2] = {
        { { -1, 0 }, { 0, 0 }, { 0, 1 }, { 1, 1 } },   // S
        { { -1, 1 }, { 0, 1 }, { 0, 0 }, { 1, 0 } },   // Z
        { { -1, 0 }, { 0, 0 }, { 1, 0 }, { -1, 1 } },  // J
        { { -1, 0 }, { 0, 0 }, { 1, 0 }, { 1, 1 } },   // L
        { { -1, 0 }, { 0, 0 }, { 1, 0 }, { 0, 1 } },   // T
        { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } },    // O
        { { -1, 0 }, { 0, 0 }, { 1, 0 }, { 2, 0 } },   // I
    };
    const PieceType order[7] = { PieceType::S, PieceType::Z, PieceType::J, PieceType::L,
                                 PieceType::T, PieceType::O, PieceType::I };
    const int t = (int)(std::find(order, order + 7, piece.type) - order);
    std::array<std::pair<int, int>, 4> out{};
    for (int k = 0; k < 4; k++) {
        const int x = north[t][k][0], y = north[t][k][1];
        int dx = x, dy = y;
        switch (piece.rotation) {
        case RotationDirection::East: dx = y; dy = -x; break;
        case RotationDirection::South: dx = -x; dy = -y; break;
        case RotationDirection::West: dx = -y; dy = x; break;
        default: break;
        }
        out[k] = { piece.position.x + dx, piece.position.y + dy };
    }
    std::sort(out.begin(), out.end());
    return out;
}

static int spinRank(const Piece& piece) {
    if (piece.type != PieceType::T) return 0;
    return piece.spin == spinType::normal ? 2 : piece.spin == spinType::mini ? 1 : 0;
}

// --canonical: Cold Clear's rank weights from integer score units, ties broken by occupied cells
// (tree_dump --canonical breaks them the same way)
static size_t canonicalSelect(const UCTNode& node) {
    struct Candidate {
        long value;
        std::array<std::pair<int, int>, 4> cells;
        int spin;
        uint32_t edge;
        float R;
    };
    std::vector<Candidate> ranked;
    for (size_t i = 0; i < node.actionCount(); i++) {
        const float R = node.stats(i).R;
        if (R <= CC_DEAD) continue;
        const Piece piece = node.move(i).piece;
        ranked.push_back({ std::lround((R - 1.0) * 10000.0), cells(piece), spinRank(piece), (uint32_t)i, R });
    }
    if (ranked.empty()) return 0;
    std::sort(ranked.begin(), ranked.end(), [](const Candidate& a, const Candidate& b) {
        if (a.value != b.value) return a.value > b.value;
        if (a.cells != b.cells) return a.cells < b.cells;
        return a.spin < b.spin;
    });
    const long worst = ranked.back().value;
    std::vector<int64_t> weights;
    int64_t total = 0;
    for (size_t rank = 0; rank < ranked.size(); rank++) {
        const int64_t e = ranked[rank].value - worst + 10;
        weights.push_back(e * e / (int64_t)(rank * rank + 1));
        total += weights.back();
    }
    const double u = cc_trace::uniform();
    double target = u * (double)total;
    size_t pick = ranked.size() - 1;
    for (size_t rank = 0; rank < ranked.size(); rank++) {
        target -= (double)weights[rank];
        if (target < 0) {
            pick = rank;
            break;
        }
    }
    cc_trace::Step step{ node.id, {}, u, (uint32_t)pick };
    for (size_t rank = 0; rank < ranked.size(); rank++) {
        step.ranked.push_back({ ranked[rank].edge, ranked[rank].R, (double)weights[rank] });
    }
    cc_trace::steps.push_back(std::move(step));
    return ranked[pick].edge;
}

// everything that decides a state's future in the CC search (no garbage, no opponent)
static std::string fingerprint(const EmulationGame& s) {
    std::string f(reinterpret_cast<const char*>(&s.game.board), sizeof(Board));
    f += (char)s.game.current_piece.type;
    f += s.game.hold ? (char)*s.game.hold : (char)127;
    for (PieceType q : s.game.queue) f += (char)q;
    f += (char)s.rng.bagiterator;
    for (PieceType b : s.rng.bag) f += (char)b;
    f.append(reinterpret_cast<const char*>(&s.rng.PPTRNG), sizeof(s.rng.PPTRNG));
    f += std::to_string(s.game.b2b) + "/" + std::to_string(s.game.combo);
    return f;
}

static std::string describe(const EmulationGame& s) {
    std::string d = std::string("current ") + pieceName(s.game.current_piece.type) + " hold " +
                    (s.game.hold ? pieceName(*s.game.hold) : "-") + " queue ";
    for (PieceType q : s.game.queue) d += pieceName(q);
    d += " pieces " + std::to_string(s.pieces);
    return d;
}

static std::vector<PieceType> bagLeft(const RNG& rng) {
    return std::vector<PieceType>(rng.bag.begin() + std::min<size_t>(rng.bagiterator, rng.bag.size()), rng.bag.end());
}

static EmulationGame fromPosition(const nlohmann::json& j) {
    EmulationGame game;
    game.reset_opponent();
    Board board{};
    const auto& rows = j["board"];
    for (int y = 0; y < (int)Board::height && y < (int)rows.size(); y++) {
        for (int x = 0; x < (int)Board::width; x++) {
            if (rows[y][x].get<int>() != 0) board.set(x, y);
        }
    }
    game.game.board = board;
    std::vector<PieceType> queue;
    for (const auto& p : j["queue"]) queue.push_back(pieceType(p.get<std::string>()));
    if (queue.size() < 1 + game.game.queue.size()) {
        std::cerr << "the position's queue needs the current piece and " << game.game.queue.size() << " more" << std::endl;
        std::exit(2);
    }
    game.game.current_piece = queue[0];
    for (size_t i = 0; i < game.game.queue.size(); i++) game.game.queue[i] = queue[i + 1];
    if (j["hold"].is_null()) game.game.hold = std::nullopt;
    else game.game.hold = pieceType(j["hold"].get<std::string>());
    game.game.b2b = j.value("b2b", 0);
    game.game.combo = j.value("combo", 0);
    std::vector<PieceType> left;
    for (const auto& p : j["bag"]) left.push_back(pieceType(p.get<std::string>()));
    game.rng.bagiterator = uint8_t(game.rng.bag.size() - left.size());
    std::copy(left.begin(), left.end(), game.rng.bag.begin() + game.rng.bagiterator);
    return game;
}

static EmulationGame fromSeed(uint64_t seed, int pieces) {
    EmulationGame game = newGame(seed, 0, 1);
    for (int p = 0; p < pieces; p++) {
        if (Shaktris::Utility::collides(game.game.board, game.game.current_piece)) break;
        const std::vector<PieceType> left = bagLeft(game.rng);
        const std::vector<Piece> placements = game.placements();
        double best = -1e18;
        int best_index = -1;
        for (size_t i = 0; i < placements.size(); i++) {
            const double total = Eval::eval_CC_standard(game.game, Move(placements[i], false), left);
            if (total > best) {
                best = total;
                best_index = (int)i;
            }
        }
        if (best_index < 0) break;
        game.set_move(Move(placements[best_index], false));
        game.play_moves();
        game.chance_move();
    }
    return game;
}

int main(int argc, const char** argv) {
    uint64_t seed = 5000;
    int pieces = 10;
    std::string position_file;
    uint64_t sims = 200;
    uint64_t search_seed = 1;
    uint64_t lockstep_seed = 1;
    int future = 40;
    bool clear_meter = false;
    for (int i = 1; i < argc; i += 2) {
        const std::string arg = argv[i];
        if (arg == "--canonical") {
            cc_trace::select_hook = canonicalSelect;
            i--;
            continue;
        }
        if (arg == "--clear-meter") {
            clear_meter = true;
            i--;
            continue;
        }
        if (i + 1 >= argc) {
            std::cerr << arg << " needs a value" << std::endl;
            return 2;
        }
        const std::string value = argv[i + 1];
        if (arg == "--seed") seed = std::stoull(value);
        else if (arg == "--pieces") pieces = std::stoi(value);
        else if (arg == "--position") position_file = value;
        else if (arg == "--sims") sims = std::stoull(value);
        else if (arg == "--search-seed") search_seed = std::stoull(value);
        else if (arg == "--lockstep-seed") lockstep_seed = std::stoull(value);
        else if (arg == "--future") future = std::stoi(value);
        else {
            std::cerr << "unknown option " << arg << std::endl;
            return 2;
        }
    }
    if (config().eval != "cc") {
        std::cerr << "nana_cctree compares against Cold Clear's eval: set NANA_CONFIG to a config with \"eval\": \"cc\"" << std::endl;
        return 2;
    }

    EmulationGame game;
    if (!position_file.empty()) {
        std::ifstream in(position_file);
        std::string line;
        if (!in || !std::getline(in, line)) {
            std::cerr << "can't read " << position_file << std::endl;
            return 2;
        }
        game = fromPosition(nlohmann::json::parse(line));
    }
    else {
        game = fromSeed(seed, pieces);
    }
    if (clear_meter) {
        game.garbage_meter.clear();
        game.buffered_garbage.clear();
        game.outgoing_garbage.clear();
    }

    fast_seed_reset((uint64_t)seedFor(search_seed, 2) | 1);
    cc_trace::seed(lockstep_seed);
    Search::simulation_limit = sims;
    Search::startSearch(game, 1);
    if (!Search::waitForSimulations(std::chrono::seconds(120))) {
        std::cerr << "warning: the search didn't reach " << sims << " simulations in 120 s" << std::endl;
    }
    Search::endSearch();
    const Move best = Search::bestMove();

    // the root as the search saw it: its future past the preview is fixed (deterministic chance,
    // which endSearch turns off)
    EmulationGame::deterministic_chance = true;
    EmulationGame root = Search::root_state;
    root.start_from_root();
    const uint32_t root_hash = root.hash();

    nlohmann::json position;
    {
        nlohmann::json rows = nlohmann::json::array();
        for (int y = 0; y < 40; y++) {
            nlohmann::json row = nlohmann::json::array();
            for (int x = 0; x < 10; x++) row.push_back(y < (int)Board::height && root.game.board.get(x, y) ? 1 : 0);
            rows.push_back(row);
        }
        position["board"] = rows;
        nlohmann::json queue = nlohmann::json::array({ pieceName(root.game.current_piece.type) });
        for (PieceType q : root.game.queue) queue.push_back(pieceName(q));
        position["preview"] = (int)queue.size();
        RNG rng = root.rng;
        for (int k = 0; k < future; k++) queue.push_back(pieceName(rng.getPiece()));
        position["queue"] = queue;
        position["hold"] = root.game.hold ? nlohmann::json(pieceName(*root.game.hold)) : nlohmann::json(nullptr);
        nlohmann::json bag = nlohmann::json::array();
        for (PieceType b : bagLeft(rng)) bag.push_back(pieceName(b));
        position["bag"] = bag;
        nlohmann::json preview_bag = nlohmann::json::array();
        for (PieceType b : bagLeft(root.rng)) preview_bag.push_back(pieceName(b));
        position["preview_bag"] = preview_bag;
        position["b2b"] = (int)root.game.b2b;
        position["combo"] = (int)root.game.combo;
        // incoming garbage: nana's search lands it and its CC eval counts it (NANA_EVAL_TANK);
        // Cold Clear's search knows nothing of it
        position["meter"] = root.garbage_meter;
        position["buffered"] = root.buffered_garbage;
    }

    // replay every select step from the root: with deterministic chance a think's path reproduces
    // the states the search saw. a node's edges belong to the state it was expanded from; a step
    // reaching the node in a different state (same hash) is a merge
    const auto& steps = cc_trace::steps;
    std::unordered_map<uint32_t, EmulationGame> created;
    created.emplace(root_hash, root);
    std::vector<bool> step_merged(steps.size(), false);
    std::vector<bool> step_lost(steps.size(), false);
    size_t merged_steps = 0;
    size_t lost_steps = 0;
    nlohmann::json merged_examples = nlohmann::json::array();
    {
        EmulationGame state = root;
        bool lost = false;
        for (size_t s = 0; s < steps.size(); s++) {
            const cc_trace::Step& step = steps[s];
            if (step.node == root_hash) {
                state = root;
                lost = false;
            }
            if (lost || state.hash() != step.node) {
                lost = true;
                step_lost[s] = true;
                lost_steps++;
                continue;
            }
            const auto it = created.find(step.node);
            if (it == created.end()) {
                created.emplace(step.node, state);
            }
            else if (fingerprint(it->second) != fingerprint(state)) {
                step_merged[s] = true;
                merged_steps++;
                if (merged_examples.size() < 5) {
                    merged_examples.push_back({ { "node", hex(step.node) }, { "expanded_from", describe(it->second) },
                                                { "reached_as", describe(state) } });
                }
            }
            UCTNode& node = Search::uct.getNode(step.node);
            state.set_move(state.specific_move(node.move(step.ranked[step.pick].edge)));
            state.play_moves();
            state.chance_move();
            const bool leaf = s + 1 == steps.size() || steps[s + 1].node == root_hash;
            const uint32_t h = state.hash();
            if (leaf && !state.game_over && Search::uct.nodeExists(h) && !created.count(h)) {
                created.emplace(h, state);
            }
        }
    }

    std::unordered_map<uint32_t, std::vector<std::string>> keys;
    nlohmann::json nodes = nlohmann::json::array();
    for (auto& [hash, state] : created) {
        UCTNode& node = Search::uct.getNode(hash);
        nlohmann::json children = nlohmann::json::array();
        std::vector<std::string>& node_keys = keys[hash];
        for (size_t i = 0; i < node.actionCount(); i++) {
            EmulationGame child = state;
            const Move m = child.specific_move(node.move(i));
            node_keys.push_back(moveKey(m.piece));
            child.set_move(m);
            child.play_moves();
            child.chance_move();
            const uint32_t child_hash = child.hash();
            const bool expanded = Search::uct.nodeExists(child_hash);
            children.push_back({
                { "mv", node_keys.back() },
                { "R", node.stats(i).R },
                { "value", ccUnits(node.stats(i).R) },
                { "static", ccUnits(node.eval(i)) },
                { "reward", std::lround(node.cold[i].move_reward * 10000.0) },
                { "N", node.stats(i).N },
                { "game_over", child.game_over },
                { "child", expanded ? nlohmann::json(hex(child_hash)) : nlohmann::json(nullptr) },
            });
        }
        nodes.push_back({
            { "id", hex(hash) },
            { "N", node.N },
            { "value", ccUnits(node.ccValue()) },
            { "children", children },
        });
    }

    nlohmann::json trace = nlohmann::json::array();
    for (size_t s = 0; s < steps.size(); s++) {
        const cc_trace::Step& step = steps[s];
        const auto it = keys.find(step.node);
        nlohmann::json ranked = nlohmann::json::array();
        for (const cc_trace::Candidate& c : step.ranked) {
            const std::string key = it != keys.end() && c.edge < it->second.size() ? it->second[c.edge] : "?";
            ranked.push_back({ key, ccUnits(c.R), c.R, c.weight });
        }
        trace.push_back({ { "node", hex(step.node) }, { "u", step.u }, { "pick", step.pick }, { "ranked", ranked },
                          { "merged", (bool)step_merged[s] }, { "lost", (bool)step_lost[s] } });
    }

    uint64_t rollouts = 0;
    for (const WorkerStatistics& s : Search::uct.stats) rollouts += s.rollouts;

    nlohmann::json out;
    out["tool"] = "nana";
    out["position"] = position;
    out["root"] = hex(root_hash);
    out["best"] = best.null_move ? nlohmann::json(nullptr) : nlohmann::json(moveKey(root.specific_move(best).piece));
    out["sims"] = sims;
    out["rollouts"] = rollouts;
    out["load_factor"] = NANA_LOAD_FACTOR;
    out["tree"] = nodes;
    out["trace"] = trace;
    out["merged_steps"] = merged_steps;
    out["merged_examples"] = merged_examples;
    out["lost_steps"] = lost_steps;
    out["unreached_nodes"] = (int)Search::uct.map_size() - (int)created.size();
    std::cout << out.dump() << std::endl;
    std::exit(0);
}
