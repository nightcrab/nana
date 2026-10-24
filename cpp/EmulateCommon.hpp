#pragma once
// what nana_emulate and nana_emulate_tbp share, so a seed deals the same game in both: its pieces,
// garbage holes and opponent

#include "EmulationGame.hpp"

#include <cstdint>

inline uint32_t seedFor(uint64_t game_seed, uint64_t stream) {
    // splitmix64 of (seed, stream), so neighbouring seeds give unrelated games
    uint64_t z = game_seed * 0x9E3779B97F4A7C15ull + stream * 0xD1B54A32D192ED03ull + 0x632BE59BD9B4E019ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return uint32_t((z ^ (z >> 31)) >> 32);
}

// ramp: pieces until the opponent's garbage is doubled (0 = no ramp)
inline EmulationGame newGame(uint64_t game_seed, int ramp, float garbage_scale) {
    EmulationGame game;
    game.garbage_ramp = ramp;
    game.garbage_scale = garbage_scale;

    game.rng.PPTRNG = seedFor(game_seed, 0);
    game.rng.makebag();

    game.reset_opponent();
    game.markov.seed(seedFor(game_seed, 1));

    game.game.current_piece = game.rng.getPiece();
    for (auto& piece : game.game.queue) {
        piece = game.rng.getPiece();
    }
    return game;
}
