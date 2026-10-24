// survival benchmark: plays seeded games against the emulated opponent with a fixed think
// time per move and reports how long the bot survives. games are reproducible per seed apart
// from the search itself (pieces, garbage holes and the opponent come from the game's seed),
// so two configs played on the same seeds can be compared game by game.
//
// usage: nana_emulate [--threads N] [--games K] [--ms T] [--seed S] [--ramp R] [--max-pieces P]
//                  [--opponent-log FILE]
//                  [--sims N] [--search-seed 0|1] [--cpu K] [--print-config] [--garbage-scale X]
//
// prints one JSON line per game to stdout and a summary to stderr.
// the games are played against MarkovOpponent, which the search models too. --opponent-log writes what passed
// between the bot and the opponent each piece, to fit MarkovOpponent with
// scripts/train_markov_opponent.py.
// --sims N searches each move for N simulations instead of --ms milliseconds. with --threads 1 and
// --search-seed 1 (the search's randomness seeded from the game seed) a game then replays exactly,
// so tuner/ can compare configs on identical games. --cpu K runs the process on logical cpu K only,
// for running games side by side. --print-config prints the NANA_CONFIG in effect as JSON and exits.
// --garbage-scale X multiplies the opponent's garbage (0.5 = half, 0 = none), for the search too.
// on Linux, run several at once on separate chiplets with e.g. taskset -c 8-15.

#include "EmulateCommon.hpp"
#include "EmulationGame.hpp"
#include "OpponentFilter.hpp"
#include "Search.hpp"
#include "UCT.hpp"
#include "Config.hpp"
#include "Util/Affinity.hpp"
#include "Util/FastSeed.hpp"
#include "engine/Utility.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#ifdef _WIN32
// windows sleeps in ~15.6 ms ticks by default, which overshoots short think times by ~6 ms
extern "C" __declspec(dllimport) unsigned __stdcall timeBeginPeriod(unsigned period);
#pragma comment(lib, "winmm")
#endif

struct Options {
    int threads = 3;
    int games = 10;
    int ms = 50;
    uint64_t seed = 0;
    // pieces until the opponent's garbage is doubled (0 = no ramp)
    int ramp = 0;
    // games still alive here are cut off (counted as survived, not died)
    int max_pieces = 2000;
    std::string opponent_log;
    // simulations per move instead of ms (0 = use ms)
    uint64_t sims = 0;
    bool search_seed = false;
    int cpu = -1;
    bool print_config = false;
    float garbage_scale = 1;
};

struct GameResult {
    int pieces = 0;
    bool died = false;
    uint64_t simulations = 0;
    // MP-MCTS: backprops that restarted below the root, reached it, or were sent on up by the leaky gate
    uint64_t restarts = 0;
    uint64_t root_backprops = 0;
    uint64_t leaks = 0;
    uint64_t illegal_moves = 0;
    uint64_t replay_matched = 0;
    uint64_t replay_diverged = 0;
    uint64_t replay_diverge_depth = 0;
    uint64_t flushes = 0;
    // NANA_FLUSH_AT_END: moves whose choice the parked buffers' hidden visits would change, and the
    // share of the root's visits that were hidden, summed over moves
    uint64_t flush_searches = 0;
    uint64_t flush_changed = 0;
    double hidden_share = 0;
    double think_seconds = 0;
    // where think time goes beyond searching: starting the workers, waking late from
    // the sleep, and stopping them (which waits for at least one finished simulation)
    double start_seconds = 0;
    double oversleep_seconds = 0;
    double stop_seconds = 0;
};

static GameResult playGame(EmulationGame& game, const Options& options, uint64_t game_seed, std::ostream* opponent_log) {
    using clock = std::chrono::steady_clock;
    auto seconds = [](clock::duration d) { return std::chrono::duration<double>(d).count(); };

    GameResult result;
    bool first = true;
    // what the search's root knows of the opponent, as under TBP1: a filter over the attacks we see
    // land (the simulated opponents have no board to encode)
    OpponentFilter opponent_filter;
    RootOpponent root_opponent;
    Search::root_opponent = &root_opponent;
    Search::root_filter = &opponent_filter;
    const Search::FlushStats flush_start = Search::flush_stats;

    while (true) {
        if (Shaktris::Utility::collides(game.game.board, game.game.current_piece)) {
            game.game_over = true;
        }
        if (game.game_over) {
            result.died = true;
            break;
        }
        if (game.ramp_pieces >= options.max_pieces) {
            break;
        }

        // with no legal placement the search can't run a simulation (its jobs end at the root), so a
        // simulation budget would wait out its timeout: it's lost, as when the search finds no move
        if (options.sims > 0 && UCTNode(game).actionCount() == 0) {
            game.game_over = true;
            continue;
        }

        // the budget is wall time per move, including starting and stopping the workers
        auto start = clock::now();
        root_opponent = opponent_filter.root();
        if (first) {
            Search::startSearch(game, options.threads);
            first = false;
        }
        else {
            Search::continueSearch(game);
        }
        auto started = clock::now();
        auto deadline = start + std::chrono::milliseconds(options.ms);
        if (options.sims > 0) {
            if (!Search::waitForSimulations(std::chrono::seconds(120))) {
                uint64_t rollouts = 0;
                for (const WorkerStatistics& s : Search::uct.stats) {
                    rollouts += s.rollouts;
                }
                std::cerr << "warning: a search didn't reach " << options.sims << " simulations in 120 s (piece "
                    << game.ramp_pieces << ", " << UCTNode(game).actionCount() << " moves at the root, "
                    << rollouts << " rollouts)" << std::endl;
            }
            deadline = started;
        }
        else {
            std::this_thread::sleep_until(deadline);
        }
        auto woke = clock::now();
        Search::endSearch();
        Move move = Search::bestMove();
        auto end = clock::now();

        result.think_seconds += seconds(end - start);
        result.start_seconds += seconds(started - start);
        result.oversleep_seconds += seconds(woke - std::max(deadline, started));
        result.stop_seconds += seconds(end - woke);

        // every simulation ends in exactly one rollout
        for (const WorkerStatistics& stats : Search::uct.stats) {
            result.simulations += stats.deferred;
            result.restarts += stats.restarts;
            result.root_backprops += stats.root_backprops;
            result.leaks += stats.leaks;
            result.illegal_moves += stats.illegal_moves;
            result.replay_matched += stats.replay_matched;
            result.replay_diverged += stats.replay_diverged;
            result.replay_diverge_depth += stats.replay_diverge_depth;
            result.flushes += stats.flushes;
        }

        if (move.null_move) {
            // nothing searched at the root: no legal placement left
            game.game_over = true;
            continue;
        }

        game.set_move(game.specific_move(move));
        int meter = 0;
        for (int garbage : game.garbage_meter) {
            meter += garbage;
        }
        const int attack_before = game.attack;
        const int lines_before = game.lines;
#if NANA_FUTURE_ORACLE
        // Verify the search root really predicts the next real transition, including RNGs.
        // In the garbage-off arm restore the real scale for this check only.
        EmulationGame prediction = Search::root_state;
        prediction.garbage_scale = game.garbage_scale;
        EmulationGame::deterministic_chance = true;
        prediction.start_from_root();
        EmulationGame::deterministic_chance = false;
        // Validate the retained pieces/opponent against reality while supplying the true hole
        // solely for this assertion, never to the partial oracle's search.
        if constexpr (NANA_ORACLE_RANDOM_HOLES) prediction.hole_rng = game.hole_rng;
        prediction.set_move(prediction.specific_move(move));
        prediction.play_moves();
        prediction.chance_move();
#endif
        game.play_moves();
        // what tbp.cpp sees: our attack, whether we cleared, and the meter we tanked if we didn't
        const int attack = game.attack - attack_before;
        const bool cleared = game.lines > lines_before;
        opponent_filter.turn(attack, cleared, cleared ? 0 : std::max(0, meter - attack));
        game.chance_move();
#if NANA_FUTURE_ORACLE
        bool exact = prediction.game.queue == game.game.queue
            && prediction.rng.PPTRNG == game.rng.PPTRNG && prediction.rng.bag == game.rng.bag
            && prediction.rng.bagiterator == game.rng.bagiterator && prediction.hole_rng == game.hole_rng
            && prediction.garbage_meter == game.garbage_meter
            && prediction.last_received == game.last_received
            && prediction.markov.last_attack == game.markov.last_attack
            && prediction.markov.deaths == game.markov.deaths;
        for (int x = 0; x < Board::width; ++x)
            exact = exact && prediction.game.board.get_column(x) == game.game.board.get_column(x);
        if (!exact) {
            std::cerr << "future oracle replay mismatch, seed " << game_seed
                      << ", piece " << game.ramp_pieces << std::endl;
            std::exit(3);
        }
#endif
        if (opponent_log) {
            game.write_opponent_step(*opponent_log, game_seed);
        }
    }

    result.pieces = game.ramp_pieces;
    result.flush_searches = Search::flush_stats.searches - flush_start.searches;
    result.flush_changed = Search::flush_stats.changed - flush_start.changed;
    result.hidden_share = Search::flush_stats.hidden_share - flush_start.hidden_share;
    return result;
}

static bool parse(int argc, const char** argv, Options& options) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--print-config") {
            options.print_config = true;
            continue;
        }
        if (i + 1 >= argc) {
            return false;
        }
        if (arg == "--opponent-log") {
            options.opponent_log = argv[++i];
            continue;
        }
        long long value = std::stoll(argv[++i]);
        if (arg == "--threads") options.threads = (int)value;
        else if (arg == "--games") options.games = (int)value;
        else if (arg == "--ms") options.ms = (int)value;
        else if (arg == "--seed") options.seed = (uint64_t)value;
        else if (arg == "--ramp") options.ramp = (int)value;
        else if (arg == "--max-pieces") options.max_pieces = (int)value;
        else if (arg == "--sims") options.sims = (uint64_t)value;
        else if (arg == "--search-seed") options.search_seed = value != 0;
        else if (arg == "--cpu") options.cpu = (int)value;
        else if (arg == "--garbage-scale") {
            options.garbage_scale = std::stof(argv[i]);
            if (options.garbage_scale < 0) return false;
        }
        else return false;
    }
    return options.threads > 0 && options.games > 0 && options.ms > 0;
}

int main(int argc, const char** argv) {
    Options options;
    try {
        if (!parse(argc, argv, options)) {
            throw std::invalid_argument("bad arguments");
        }
    }
    catch (const std::exception&) {
        std::cerr << "usage: " << argv[0]
            << " [--threads N] [--games K] [--ms T] [--seed S] [--ramp R] [--max-pieces P]"
            << " [--opponent-log FILE] [--sims N] [--search-seed 0|1] [--cpu K] [--print-config] [--garbage-scale X]"
            << std::endl;
        return 1;
    }

    if (options.print_config) {
        std::cout << configJson(config()) << std::endl;
        return 0;
    }
    if (options.cpu >= 0 && !restrictProcessToCpu(options.cpu)) {
        std::cerr << "can't restrict the process to cpu " << options.cpu << std::endl;
        return 1;
    }
    Search::simulation_limit = options.sims;

#ifdef _WIN32
    timeBeginPeriod(1);
#endif

    std::ofstream opponent_log;
    if (!options.opponent_log.empty()) {
        opponent_log.open(options.opponent_log);
        if (!opponent_log) {
            std::cerr << "can't write " << options.opponent_log << std::endl;
            return 1;
        }
        EmulationGame::write_opponent_header(opponent_log);
    }

    int deaths = 0;
    long long total_pieces = 0;
    uint64_t total_simulations = 0;
    double total_think = 0;

    for (int i = 0; i < options.games; i++) {
        uint64_t game_seed = options.seed + i;
        // before the game's state exists: constructing it draws seeds too
        fast_seed_reset(options.search_seed ? (uint64_t)seedFor(game_seed, 2) | 1 : 0);
        EmulationGame game = newGame(game_seed, options.ramp, options.garbage_scale);
        GameResult result = playGame(game, options, game_seed, opponent_log ? &opponent_log : nullptr);

        deaths += result.died;
        total_pieces += result.pieces;
        total_simulations += result.simulations;
        total_think += result.think_seconds;

        int moves = std::max(result.pieces, 1);
        std::cout << "{\"seed\":" << game_seed
            << ",\"rules\":\"" << EmulationGame::rules() << "\""
            << ",\"threads\":" << options.threads
            << ",\"ms\":" << options.ms
            << ",\"ramp\":" << options.ramp
            << ",\"max_pieces\":" << options.max_pieces
            << ",\"pieces\":" << result.pieces
            << ",\"died\":" << (result.died ? "true" : "false")
            << ",\"attack\":" << game.attack
            << ",\"true_attack\":" << game.true_attack
            << ",\"lines\":" << game.lines
            // before the ramp scales it
            << ",\"opponent_attack\":" << game.markov.total_damage
            << ",\"opponent_deaths\":" << game.markov.deaths
            << ",\"sims_per_move\":" << result.simulations / moves
            << ",\"restarts_per_sim\":" << (double)result.restarts / std::max<uint64_t>(result.simulations, 1)
            << ",\"root_per_sim\":" << (double)result.root_backprops / std::max<uint64_t>(result.simulations, 1)
            << ",\"leaks_per_sim\":" << (double)result.leaks / std::max<uint64_t>(result.simulations, 1)
            << ",\"illegal_per_sim\":" << (double)result.illegal_moves / std::max<uint64_t>(result.simulations, 1)
            << ",\"replay_match_rate\":" << (double)result.replay_matched / std::max<uint64_t>(result.replay_matched + result.replay_diverged, 1)
            << ",\"replay_diverge_depth\":" << (double)result.replay_diverge_depth / std::max<uint64_t>(result.replay_diverged, 1)
            << ",\"flushes_per_sim\":" << (double)result.flushes / std::max<uint64_t>(result.simulations, 1)
            << ",\"flush_changed\":" << (double)result.flush_changed / std::max<uint64_t>(result.flush_searches, 1)
            << ",\"hidden_share\":" << result.hidden_share / std::max<uint64_t>(result.flush_searches, 1)
            << ",\"ms_per_move\":" << 1000.0 * result.think_seconds / moves
            << ",\"start_ms\":" << 1000.0 * result.start_seconds / moves
            << ",\"oversleep_ms\":" << 1000.0 * result.oversleep_seconds / moves
            << ",\"stop_ms\":" << 1000.0 * result.stop_seconds / moves
            << "}" << std::endl;
    }

    // with cut-off games, pieces / deaths is the maximum-likelihood mean survival
    // (exponential lifetimes); without any it is just the mean
    std::cerr << EmulationGame::rules() << " rules, games " << options.games << ", died " << deaths
        << ", mean pieces " << (double)total_pieces / options.games
        << ", pieces per death " << (deaths ? (double)total_pieces / deaths : 0.0)
        << ", sims/move " << (total_pieces ? total_simulations / total_pieces : 0)
        << ", ms/move " << (total_pieces ? 1000.0 * total_think / total_pieces : 0.0) << std::endl;
    return 0;
}
