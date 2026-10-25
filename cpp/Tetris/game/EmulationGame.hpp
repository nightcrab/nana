#pragma once
#include "MarkovOpponent.hpp"
#include "OpponentFilter.hpp"
#include "engine/Game.hpp"
#include "util/rng.hpp"

#include <ostream>
#include <queue>

#ifndef NANA_HASH_QUEUE_FRONT
#define NANA_HASH_QUEUE_FRONT 1
#endif

#ifndef NANA_FUTURE_ORACLE
#define NANA_FUTURE_ORACLE 0
#endif
#ifndef NANA_SEPARATE_HOLES
#define NANA_SEPARATE_HOLES 0
#endif
#ifndef NANA_ORACLE_RANDOM_HOLES
#define NANA_ORACLE_RANDOM_HOLES 0
#endif
static_assert(!NANA_ORACLE_RANDOM_HOLES || (NANA_FUTURE_ORACLE && NANA_SEPARATE_HOLES),
              "random-hole oracle requires future oracle and separate hole RNG");

// analogous to an APM survival mode
class EmulationGame {
   public:
    explicit EmulationGame(bool apply_rules = true);
    static const char* rules();
    static int default_garbage_delay();
    static void set_garbage_delay(int delay);

    Game game;
#ifdef NANA_SEARCH_FIXTURE
    bool fixture_active = false;
    int fixture_node = 0;
    int fixture_action = 0;
#endif

    int garbage_delay = 0;
    std::vector<int> buffered_garbage;
    std::vector<int> outgoing_garbage;
    std::vector<int> garbage_meter;
    bool phantom_garbage = false;
    std::vector<int> phantom_meter;

    // can be set if the opponent already has some B2B
    float multiplier = 1.0;

    // can use below if we know opponent is combo'ing and likely to extend
    // opponent's existing combo
    int combo = 0;
    // longest combo extension possible
    int combo_length = 0;

    Move move;
    RNG rng;
    uint32_t hole_rng = 0;
    int next_hole() {
        if constexpr (NANA_SEPARATE_HOLES) {
            hole_rng = hole_rng * 0x5d588b65 + 0x269ec3;
            return ((hole_rng >> 16) * Board::width) >> 16;
        }
        return rng.getRand(Board::width);
    }
    MarkovOpponent markov;
    bool track_belief = false;
    OpponentFilter belief;

    void set_tetrio_rules();

    float kill_value = 0;
    float kill_survival = 1;
    float kill_weight = 1;
    float kill_discount = 1;
    void reset_kill(float discount, bool expected);
    float expected_kill() const;

    int garbage_ramp = 0;
    int ramp_pieces = 0;
    float ramp_carry = 0;
    float garbage_scale = 1;

    int pieces = 0;
    int lines = 0;
    int attack = 0;

    // does not count cancelling
    int true_attack = 0;

    int last_sent = 0;
    int last_received = 0;

    bool game_over = false;

    float app();
    float true_app();
    float apl();

    float b2b();

    void set_move(Move move);
    Move specific_move(Move move);

    // Opp is MarkovOpponent or NoOpponent; the search picks one per run and the environment always uses the default
    template <class Opp = MarkovOpponent>
    void chance_move();

    template <class Opp = MarkovOpponent>
    void play_moves();
    void objectively_play_moves();

    std::vector<Piece> placements() const;
    std::vector<Move> legal_moves();

    uint32_t hash() const;

    double opponent_jeopardy() const;
    double opponent_deaths() const;
    bool opponent_dead() const;
    void reset_opponent_rng();
    template <class Opp = MarkovOpponent>
    void reseed_chance();
    static inline bool deterministic_chance = false;
    template <class Opp = MarkovOpponent>
    void start_from_root();
    void reset_opponent();

    static void write_opponent_header(std::ostream& out);
    void write_opponent_step(std::ostream& out, uint64_t game_id) const;
};
