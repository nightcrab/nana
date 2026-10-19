// survival benchmark: plays seeded games against the emulated opponent with a fixed think
// time per move and reports how long the bot survives. games are reproducible per seed apart
// from the search itself (pieces, garbage holes and the opponent come from the game's seed),
// so two configs played on the same seeds can be compared game by game.
//
// usage: nana_play [--threads N] [--games K] [--ms T] [--seed S] [--ramp R] [--max-pieces P]
//
// prints one JSON line per game to stdout and a summary to stderr.
// on Linux, run several at once on separate chiplets with e.g. taskset -c 8-15.

#include "EmulationGame.hpp"
#include "Search.hpp"
#include "engine/Utility.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
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
};

static uint32_t seedFor(uint64_t game_seed, uint64_t stream) {
    // splitmix64 of (seed, stream), so neighbouring seeds give unrelated games
    uint64_t z = game_seed * 0x9E3779B97F4A7C15ull + stream * 0xD1B54A32D192ED03ull + 0x632BE59BD9B4E019ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return uint32_t((z ^ (z >> 31)) >> 32);
}

static EmulationGame newGame(uint64_t game_seed, int ramp) {
    EmulationGame game;
    game.garbage_ramp = ramp;

    game.rng.PPTRNG = seedFor(game_seed, 0);
    game.rng.makebag();

    // the opponent draws its first queue state at construction, from an unseeded rng
    game.opponent.rng.PPTRNG = seedFor(game_seed, 1);
    game.opponent.nextI = game.opponent.rng.getRand(7);
    game.opponent.nextSpinPiece = game.opponent.rng.getRand(7);

    game.game.current_piece = game.rng.getPiece();
    for (auto& piece : game.game.queue) {
        piece = game.rng.getPiece();
    }
    return game;
}

struct GameResult {
    int pieces = 0;
    bool died = false;
    uint64_t simulations = 0;
    double think_seconds = 0;
    // where think time goes beyond searching: starting the workers, waking late from
    // the sleep, and stopping them (which waits for at least one finished simulation)
    double start_seconds = 0;
    double oversleep_seconds = 0;
    double stop_seconds = 0;
};

static GameResult playGame(EmulationGame& game, const Options& options) {
    using clock = std::chrono::steady_clock;
    auto seconds = [](clock::duration d) { return std::chrono::duration<double>(d).count(); };

    GameResult result;
    bool first = true;

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

        // the budget is wall time per move, including starting and stopping the workers
        auto start = clock::now();
        if (first) {
            Search::startSearch(game, options.threads);
            first = false;
        }
        else {
            Search::continueSearch(game);
        }
        auto started = clock::now();
        auto deadline = start + std::chrono::milliseconds(options.ms);
        std::this_thread::sleep_until(deadline);
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
        }

        if (move.null_move) {
            // nothing searched at the root: no legal placement left
            game.game_over = true;
            continue;
        }

        game.set_move(game.specific_move(move));
        game.play_moves();
        game.chance_move();
    }

    result.pieces = game.ramp_pieces;
    return result;
}

static bool parse(int argc, const char** argv, Options& options) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (i + 1 >= argc) {
            return false;
        }
        long long value = std::stoll(argv[++i]);
        if (arg == "--threads") options.threads = (int)value;
        else if (arg == "--games") options.games = (int)value;
        else if (arg == "--ms") options.ms = (int)value;
        else if (arg == "--seed") options.seed = (uint64_t)value;
        else if (arg == "--ramp") options.ramp = (int)value;
        else if (arg == "--max-pieces") options.max_pieces = (int)value;
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
            << " [--threads N] [--games K] [--ms T] [--seed S] [--ramp R] [--max-pieces P]" << std::endl;
        return 1;
    }

#ifdef _WIN32
    timeBeginPeriod(1);
#endif

    int deaths = 0;
    long long total_pieces = 0;
    uint64_t total_simulations = 0;
    double total_think = 0;

    for (int i = 0; i < options.games; i++) {
        uint64_t game_seed = options.seed + i;
        EmulationGame game = newGame(game_seed, options.ramp);
        GameResult result = playGame(game, options);

        deaths += result.died;
        total_pieces += result.pieces;
        total_simulations += result.simulations;
        total_think += result.think_seconds;

        int moves = std::max(result.pieces, 1);
        std::cout << "{\"seed\":" << game_seed
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
            << ",\"opponent_attack\":" << game.opponent.total_damage
            << ",\"sims_per_move\":" << result.simulations / moves
            << ",\"ms_per_move\":" << 1000.0 * result.think_seconds / moves
            << ",\"start_ms\":" << 1000.0 * result.start_seconds / moves
            << ",\"oversleep_ms\":" << 1000.0 * result.oversleep_seconds / moves
            << ",\"stop_ms\":" << 1000.0 * result.stop_seconds / moves
            << "}" << std::endl;
    }

    // with cut-off games, pieces / deaths is the maximum-likelihood mean survival
    // (exponential lifetimes); without any it is just the mean
    std::cerr << "games " << options.games << ", died " << deaths
        << ", mean pieces " << (double)total_pieces / options.games
        << ", pieces per death " << (deaths ? (double)total_pieces / deaths : 0.0)
        << ", sims/move " << (total_pieces ? total_simulations / total_pieces : 0)
        << ", ms/move " << (total_pieces ? 1000.0 * total_think / total_pieces : 0.0) << std::endl;
    return 0;
}
