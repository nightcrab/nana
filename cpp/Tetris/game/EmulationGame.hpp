#pragma once
#include "Opponent.hpp"
#include "MarkovOpponent.hpp"
#include "engine/Game.hpp"

#include <ostream>
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
    // NANA_RULES=tetrio plays TETR.IO rules (set_tetrio_rules); otherwise Botris damage and the
    // original garbage rules
    EmulationGame();

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
    // plays markov in place of opponent
    MarkovOpponent markov;
    bool markov_opponent = false;

    // the stadium's garbage rules (Universal-Shaktris-Stadium's VersusGame): the opponent's attack
    // reaches the meter the same turn instead of a turn later, and a piece that clears no lines
    // tanks the whole meter at once (one hole), not 8 lines
    bool stadium_rules = false;
    // TETR.IO damage (TetrioS1, the same table as the stadium's TetrioStats) and stadium_rules
    void set_tetrio_rules();

    // the opponent's death along the search path, for the search's reward: kill_value is
    // sum over its pieces t of P(alive before t) P(dies at t) discount^t, kill_survival P(alive now)
    // and kill_weight discount^(pieces so far). the HMM gives P(dies at t) exactly (expected_death);
    // the handcrafted opponent's is 0 or 1
    float kill_value = 0;
    float kill_survival = 1;
    float kill_weight = 1;
    float kill_discount = 1;
    // start counting from here; expected: the HMM plays on as if it survived and reports the odds
    void reset_kill(float discount, bool expected);
    // the expected discounted kill, from the path and the HMM's death_value from here on
    float expected_kill() const;

    // survival-mode garbage ramp: the opponent's garbage is scaled by
    // 1 + ramp_pieces / garbage_ramp (0 = off). the search plays the same ramp
    int garbage_ramp = 0;
    // pieces placed this game; unlike pieces, the search doesn't reset it at each root
    int ramp_pieces = 0;
    // fraction of a line owed from scaling, so the ramp is exact without using the rng
    float ramp_carry = 0;
    // the opponent's garbage times this (1 = as sent), before the ramp. tuner/ eases it for
    // searches too weak to survive full pressure; the search plays the same scale
    float garbage_scale = 1;

    int pieces = 0;
    int lines = 0;
    int attack = 0;

    // does not count cancelling
    int true_attack = 0;

    // damage sent to the opponent by the last move, after cancelling, for logging
    int last_sent = 0;
    // damage the opponent sent in the last chance_move, after the ramp, for logging
    int last_received = 0;

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

    // the opponent model in play
    double opponent_jeopardy() const;
    double opponent_deaths() const;
    bool opponent_dead() const;
    void reset_opponent_rng();
    // a fresh opponent of either model
    void reset_opponent(bool markov);

    // one CSV row of what passed between us and the opponent this piece, after play_moves and
    // chance_move: the training data for scripts/train_markov_opponent.py
    static void write_opponent_header(std::ostream& out);
    void write_opponent_step(std::ostream& out, uint64_t game_id) const;
};