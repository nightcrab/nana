#pragma once
#include "Opponent.hpp"
#include "engine/Game.hpp"

#include <queue>

// node hashes include the next piece when nothing is held: holding then plays that piece, so it
// decides the node's moves. 0 = leave it out (the original hash, under which one node can stand for
// states with different hold placements)
#ifndef NANA_HASH_QUEUE_FRONT
#define NANA_HASH_QUEUE_FRONT 1
#endif

// analogous to an APM survival mode
// simulates a versus game with a highly simplified opponent model
class EmulationGame {
   public:
    Game game;

    std::vector<int> buffered_garbage;
    std::vector<int> garbage_meter;

    // can be set if the opponent already has some B2B
    float multiplier = 1.0;

    // can use below if we know opponent is combo'ing and likely to extend
    // opponent's existing combo
    int combo = 0;
    // longest combo extension possible
    int combo_length = 0;

    Move move;
    RNG rng;
    Opponent opponent;

    // survival-mode garbage ramp: the opponent's garbage is scaled by
    // 1 + ramp_pieces / garbage_ramp (0 = off). the search plays the same ramp
    int garbage_ramp = 0;
    // pieces placed this game; unlike pieces, the search doesn't reset it at each root
    int ramp_pieces = 0;
    // fraction of a line owed from scaling, so the ramp is exact without using the rng
    float ramp_carry = 0;

    int pieces = 0;
    int lines = 0;
    int attack = 0;

    // does not count cancelling
    int true_attack = 0;

    bool game_over = false;

    float app();
    float true_app();
    float apl();

    float b2b();

    void set_move(Move move);
    Move specific_move(Move move);

    void chance_move();

    void play_moves();
    void objectively_play_moves();

    std::vector<Move> legal_moves();

    uint32_t hash() const;
};