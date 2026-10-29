#pragma once
#include "IOHMM.hpp"
#include "Neural.hpp"
#include "Belief.hpp"
#include "engine/Game.hpp"
#include "util/rng.hpp"

#include <ostream>
#include <queue>
#include <type_traits>

// Search tree key: a hash of the position. 64 bits keeps collisions between live nodes negligible.
using NodeKey = uint64_t;

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
    // The opponent's side of the garbage exchange. The opponent models only predict its attacks; the rules for
    // cancelling, its meter and its deaths are applied here.
    struct OpponentState {
        float meter = 0;
        // garbage it received since its last piece
        float received = 0;
        double last_attack = 0;
        bool dead = false;
        double deaths = 0;
        float pieces = 0;
        float total_damage = 0;
        opponent::Rng rng;
    };
    OpponentState opponent;
    opponent::IOHMM iohmm;
    opponent::Neural neural;
    bool track_belief = false;
    opponent::Belief belief;

    void set_tetrio_rules();

    int garbage_ramp = 0;
    int ramp_pieces = 0;
    float ramp_carry = 0;
    float garbage_scale = 1;

    int pieces = 0;
    int lines = 0;
    int game_score = 0;
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

    // Opp is opponent::IOHMM, opponent::Neural or opponent::None; the search picks one per run and the environment
    // always uses the default
    template <class Opp = opponent::IOHMM>
    void chance_move();

    template <class Opp = opponent::IOHMM>
    void play_moves();
    void objectively_play_moves();

    std::vector<Piece> placements() const;
    std::vector<Move> legal_moves();

    NodeKey hash() const;

    // The opponent's attack this piece after cancelling against its meter, and the garbage it receives from us.
    template <class Opp>
    double opponent_attack();
    void opponent_receive(double amount);
    void adopt_opponent(const opponent::Root& root);

    double opponent_meter() const;
    double opponent_deaths() const;
    bool opponent_dead() const;
    template <class Opp = opponent::IOHMM>
    void reseed_chance();
    static inline bool deterministic_chance = false;
    template <class Opp = opponent::IOHMM>
    void start_from_root();
    void reset_opponent();

    static void write_opponent_header(std::ostream& out);
    void write_opponent_step(std::ostream& out, uint64_t game_id) const;

private:
    template <class Opp>
    Opp& model() {
        if constexpr (std::is_same_v<Opp, opponent::IOHMM>) {
            return iohmm;
        }
        else {
            static_assert(std::is_same_v<Opp, opponent::Neural>);
            return neural;
        }
    }
};
