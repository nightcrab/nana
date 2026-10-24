#pragma once
#include "MarkovOpponent.hpp"
#include "engine/Game.hpp"
#include "util/rng.hpp"

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
    // plays TETR.IO rules (set_tetrio_rules), or with NANA_RULES=botris Botris damage and the
    // original garbage rules
    // Search placeholders skip rule loading before the executable selects its configuration.
    explicit EmulationGame(bool apply_rules = true);
    // the ruleset games are played with: "tetrio" or "botris" (from NANA_RULES, or TBP configuration)
    static const char* rules();
    // the garbage delay new games start with: TBP rules' garbage_delay (set_garbage_delay), else
    // NANA_GARBAGE_DELAY, else the ruleset's (0 for tetrio, 1 for botris)
    static int default_garbage_delay();
    static void set_garbage_delay(int delay);

    Game game;

    // turns an attack spends in transit before it joins the receiver's meter, as in the stadium's
    // VersusGame: 0 lands it the turn it's sent. under TETR.IO rules our attacks wait as long, and
    // garbage in transit can be cancelled; botris (default 1) only delays the opponent's
    int garbage_delay = 0;
    // the opponent's attacks in transit: [i] joins the meter i + 1 chance_moves from now
    std::vector<int> buffered_garbage;
    // ours (TETR.IO rules): [i] reaches the opponent i + 1 chance_moves from now, after its piece
    std::vector<int> outgoing_garbage;
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
    MarkovOpponent markov;

    // TETR.IO damage (TetrioS1, the same table as the stadium's TetrioStats), and with it the
    // stadium's rules (Universal-Shaktris-Stadium's VersusGame): only T-spins, scored as the
    // stadium validates them (see placements), an attack cancels the meter and then garbage in
    // transit, and a piece that clears no lines tanks the whole meter at once (one hole), not 8 lines
    void set_tetrio_rules();

    // the opponent's death along the search path, for the search's reward: kill_value is
    // sum over its pieces t of P(alive before t) P(dies at t) discount^t, kill_survival P(alive now)
    // and kill_weight discount^(pieces so far). the HMM gives P(dies at t) exactly (expected_death)
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

    // the placements of the current and hold pieces. under TETR.IO rules a T's spin is the one the
    // stadium grants (its Game::validate_spin): the best a rotation onto the placement from a state
    // reachable from spawn scores by the three-corner rule, and other pieces don't spin
    std::vector<Piece> placements() const;
    std::vector<Move> legal_moves();

    uint32_t hash() const;

    // the opponent model in play
    double opponent_jeopardy() const;
    double opponent_deaths() const;
    bool opponent_dead() const;
    void reset_opponent_rng();
    // fresh chance for a simulation: a new seed, and the pieces left in the current bag in a new
    // order (which pieces are left is known under seven-bag; their order isn't)
    void reseed_chance();
    // a simulation starting from the search root: fresh randomness, and the HMM opponent's hidden
    // state drawn from the root belief. on a copy of the root, never on the shared root itself
    void start_from_root();
    // a fresh HMM opponent
    void reset_opponent();

    // one CSV row of what passed between us and the opponent this piece, after play_moves and
    // chance_move: the training data for scripts/train_markov_opponent.py
    static void write_opponent_header(std::ostream& out);
    void write_opponent_step(std::ostream& out, uint64_t game_id) const;
};