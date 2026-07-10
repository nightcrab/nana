#pragma once
#include "Opponent.hpp"
#include "engine/Game.hpp"

#include <queue>

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

    int pieces = 0;
    int lines = 0;
    int attack = 0;

    // does not count cancelling
    int true_attack = 0;

    bool game_over = false;

    float app() const;
    float true_app() const;
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