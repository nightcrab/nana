#include "EmulationGame.hpp"
#include "Config.hpp"
#include "engine/RotationSystems.hpp"
#include "engine/Utility.hpp"

#include "Util/fasthash.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>

const char* EmulationGame::rules() {
    if (const auto* c = tbpConfig()) return c->rules.c_str();
    static const char* const name = [] {
        const char* rules = std::getenv("NANA_RULES");
        if (!rules || !*rules || std::strcmp(rules, "tetrio") == 0) {
            return "tetrio";
        }
        if (std::strcmp(rules, "botris") == 0) {
            return "botris";
        }
        std::cerr << "NANA_RULES=" << rules << " isn't a ruleset (tetrio or botris)" << std::endl;
        std::exit(1);
    }();
    return name;
}

namespace {
// -1 until TBP rules set it
int tbp_garbage_delay = -1;
}

int EmulationGame::default_garbage_delay() {
    if (tbp_garbage_delay >= 0) return tbp_garbage_delay;
    static const int env = [] {
        const char* delay = std::getenv("NANA_GARBAGE_DELAY");
        if (!delay || !*delay) {
            return -1;
        }
        char* end;
        const long value = std::strtol(delay, &end, 10);
        if (*end || value < 0) {
            std::cerr << "NANA_GARBAGE_DELAY=" << delay << " isn't a number of turns" << std::endl;
            std::exit(1);
        }
        return (int)value;
    }();
    if (env >= 0) return env;
    return std::strcmp(rules(), "botris") == 0 ? 1 : 0;
}

void EmulationGame::set_garbage_delay(int delay) {
    tbp_garbage_delay = delay;
}

EmulationGame::EmulationGame(bool apply_rules) {
    if (apply_rules) {
        if (std::strcmp(rules(), "tetrio") == 0) {
            set_tetrio_rules();
        }
        garbage_delay = default_garbage_delay();
    }
}

void EmulationGame::set_tetrio_rules() {
    game.mode = TetrioS1();
}

void EmulationGame::set_move(Move move) {
    this->move = move;
};

Move EmulationGame::specific_move(Move move) {
    move.piece.position.y += game.board.get_garbage_height();
    return move;
};

void EmulationGame::chance_move() {
    for (auto& piece : game.queue) {
        if (piece == PieceType::Empty) {
            piece = rng.getPiece();
        }
    }

    const bool tetrio = std::holds_alternative<TetrioS1>(game.mode);

    // the opponent's attack that has waited out the delay joins the meter
    if (garbage_delay > 0) {
        buffered_garbage.resize(garbage_delay);
        if (buffered_garbage.front() > 0) {
            garbage_meter.insert(garbage_meter.begin(), buffered_garbage.front());
        }
        std::shift_left(buffered_garbage.begin(), buffered_garbage.end(), 1);
        buffered_garbage.back() = 0;
    }

    int garbage_amount = 0;

    // simulate opponent

    garbage_amount = (int)markov.play();
    if (phantom_garbage && garbage_amount > 0) {
        phantom_meter.insert(phantom_meter.begin(), garbage_amount);
    }

    // ours that has waited out the delay reaches it, after the piece it just played (so the stadium
    // rules' tank in receiveAttack goes by that piece). what's left of its attack after its meter
    // first cancels ours in transit, soonest first
    if (tetrio && garbage_delay > 0) {
        outgoing_garbage.resize(garbage_delay);
        for (int& garbage : outgoing_garbage) {
            const int cancelled = std::min(garbage_amount, garbage);
            garbage -= cancelled;
            garbage_amount -= cancelled;
        }
        if (outgoing_garbage.front() > 0) {
            markov.receiveAttack(outgoing_garbage.front());
        }
        std::shift_left(outgoing_garbage.begin(), outgoing_garbage.end(), 1);
        outgoing_garbage.back() = 0;
    }

    if (garbage_scale != 1 && garbage_amount > 0) {
        float scaled = garbage_amount * garbage_scale + ramp_carry;
        garbage_amount = (int)scaled;
        ramp_carry = scaled - garbage_amount;
    }

    if (garbage_ramp > 0 && garbage_amount > 0) {
        float scaled = garbage_amount * (1.0f + (float)ramp_pieces / garbage_ramp) + ramp_carry;
        garbage_amount = (int)scaled;
        ramp_carry = scaled - garbage_amount;
    }

    last_received = garbage_amount;

    const float death = markov.expected_death ? markov.last_death_prob : (markov.dead ? 1.0f : 0.0f);
    kill_weight *= kill_discount;
    kill_value += kill_survival * death * kill_weight;
    kill_survival *= 1.0f - death;

    if (garbage_amount > 0) {
        if (garbage_delay == 0) {
            // both players move at once: it meets our next piece's attack, and is tanked after it
            garbage_meter.insert(garbage_meter.begin(), garbage_amount);
        }
        else {
            buffered_garbage.back() = garbage_amount;
        }
    }
};

void EmulationGame::play_moves(){
    last_sent = 0;

    if (game_over) {
        return;
    }

    if (Shaktris::Utility::collides(game.board, game.current_piece)) {
        game_over = true;
        return;
    }

    // game is not over, do a null move if we have one
    if (move.null_move) {
        return;
    }

    // but we need to keep track of combo and b2b because cancelling

    // TETR.IO rewards only T-spins; move generation also flags other pieces' spins.
    spinType spin = move.piece.spin;
    if (std::holds_alternative<TetrioS1>(game.mode) && move.piece.type != PieceType::T) {
        spin = spinType::null;
    }
    if (!move.null_move) {
        game.place_piece(move.piece);
        pieces += 1;
        ramp_pieces += 1;
    }
    int cleared_lines = game.board.clearLines();

    lines += cleared_lines;

    bool pc = true;
    for (int i = 0; i < Board::width; i++) {
        if (game.board.get_column(i) != 0) {
            pc = false;
        }
    }

    // you can't pc by doing nothing lol
    if (move.null_move) {
        pc = false;
    }


    int damage = game.damage_sent(cleared_lines, spin, pc);
    // what the belief sees: our attack before cancelling, and what then lands
    const int sent = damage;
    int landed = 0;

    attack += damage;

    // cancel damage
    while (damage > 0 && garbage_meter.size() > 0) {
        int &incoming = garbage_meter.back();

        if (incoming >= damage) {
            incoming -= damage;
            damage = 0;
        } else {
            damage -= incoming;
            incoming = 0;
        }

        if (incoming == 0) {
            garbage_meter.pop_back();
        }
    }

    const bool tetrio = std::holds_alternative<TetrioS1>(game.mode);

    if (phantom_garbage) {
        // what's left of our attack cancels the phantom meter as it would the real one, without
        // taking anything from what we send
        int left = damage;
        while (left > 0 && !phantom_meter.empty()) {
            int& incoming = phantom_meter.back();
            const int cancelled = std::min(incoming, left);
            incoming -= cancelled;
            left -= cancelled;
            if (incoming == 0) {
                phantom_meter.pop_back();
            }
        }
        if (cleared_lines == 0) {
            phantom_meter.clear();
        }
    }

    // zero pass-through: then the garbage still in transit to us, soonest first
    if (tetrio) {
        for (int& incoming : buffered_garbage) {
            const int cancelled = std::min(damage, incoming);
            incoming -= cancelled;
            damage -= cancelled;
        }
    }

    // send damage
    if (tetrio && garbage_delay > 0) {
        outgoing_garbage.resize(garbage_delay);
        outgoing_garbage.back() += damage;
    }
    else {
        markov.receiveAttack(damage);
    }
    last_sent = damage;

    // combo block
    if (cleared_lines == 0 && tetrio) {
        // the whole meter at once, one hole
        int total = 0;
        for (int garbage : garbage_meter) {
            total += garbage;
        }
        if (total > 0) {
            game.add_garbage(total, next_hole());
        }
        landed = total;
        garbage_meter.clear();
    }
    else if (cleared_lines == 0) {

        int garbage_used = 0;

        // place garbage 
        while (garbage_meter.size() > 0) {
            int& incoming = garbage_meter.back();

            int garbage = std::min(incoming, 8 - garbage_used);

            game.add_garbage(garbage, next_hole());

            incoming -= garbage;
            garbage_used += garbage;

            if (incoming == 0) {
                garbage_meter.pop_back();
            }

            if (garbage_used == 8) {
                break;
            }
        }
    }

    true_attack += damage;

    if (track_belief && tetrio) {
        belief.turn(sent, cleared_lines > 0, landed);
    }

    //game.app = app();

};

void EmulationGame::objectively_play_moves() {


}


namespace {
// best spin (spinType) a rotation lands each T state with: [rotation][x][y] of its centre
using SpinTable = std::array<std::array<std::array<uint8_t, Board::height>, Board::width>, 4>;

// the stadium's Game::validate_spin for every T placement at once: every state reachable from spawn
// by shift, SRS rotation and sonic drop, and the three-corner spin (srs_rotate, the same code as the
// stadium's Game::rotate) of each rotation that succeeds, the best kept per state
void t_spins(const Board& board, SpinTable& best) {
    const Piece spawn(PieceType::T);
    if (Shaktris::Utility::collides(board, spawn)) {
        return;
    }
    // a T's centre is one of its minos, so every state's is on the board
    std::array<std::array<std::array<bool, Board::height>, Board::width>, 4> visited{};
    std::vector<Piece> open;
    open.reserve(128);
    auto visit = [&](const Piece& piece) {
        bool& seen = visited[piece.rotation][piece.position.x][piece.position.y];
        if (!seen) {
            seen = true;
            open.push_back(piece);
        }
    };
    visit(spawn);
    while (!open.empty()) {
        const Piece piece = open.back();
        open.pop_back();
        for (TurnDirection dir : { TurnDirection::Left, TurnDirection::Right }) {
            Piece next = piece;
            next.spin = spinType::null;
            srs_rotate(board, next, dir);
            // a failed rotation leaves the piece where it was
            if (next.rotation == piece.rotation) {
                continue;
            }
            uint8_t& spin = best[next.rotation][next.position.x][next.position.y];
            spin = std::max(spin, (uint8_t)next.spin);
            visit(next);
        }
        for (int dir : { -1, 1 }) {
            Piece next = piece;
            Shaktris::Utility::shift(board, next, dir);
            visit(next);
        }
        Piece next = piece;
        Shaktris::Utility::sonic_drop(board, next);
        visit(next);
    }
}

// a spin needs three of a T's corners filled (srs_rotate). off the board counts as filled here, so
// this never misses one
bool spin_corners(const Board& board, const Piece& piece) {
    int filled = 0;
    for (int dx : { -1, 1 }) {
        for (int dy : { -1, 1 }) {
            const int x = piece.position.x + dx;
            const int y = piece.position.y + dy;
            filled += x < 0 || x >= (int)Board::width || y < 0 || y >= (int)Board::height || board.get(x, y);
        }
    }
    return filled >= 3;
}
}

std::vector<Piece> EmulationGame::placements() const {
    std::vector<Piece> pieces = game.get_possible_piece_placements();
    if (!std::holds_alternative<TetrioS1>(game.mode)) {
        // Botris rewards any piece's spin: the movegen's (immobile placements)
        return pieces;
    }

    // the movegen marks immobile placements as spins; the stadium only grants T-spins, by rotation
    bool candidates = false;
    for (Piece& piece : pieces) {
        piece.spin = spinType::null;
        candidates |= piece.type == PieceType::T && spin_corners(game.board, piece);
    }
    if (!candidates) {
        return pieces;
    }

    SpinTable best{};
    t_spins(game.board, best);
    for (Piece& piece : pieces) {
        // (the movegen occasionally yields a placement off the board: never reachable, so no spin)
        if (piece.type == PieceType::T && piece.position.x >= 0 && piece.position.x < (int)Board::width &&
            piece.position.y >= 0 && piece.position.y < (int)Board::height) {
            piece.spin = (spinType)best[piece.rotation][piece.position.x][piece.position.y];
        }
    }
    return pieces;
}

std::vector<Move> EmulationGame::legal_moves() {
    std::vector<Piece> raw_actions = placements();

    std::vector<Move> moves;

    for (auto& raw_action : raw_actions) {
        //moves.push_back(Move(raw_action, true));
        moves.push_back(Move(raw_action, false));
    }

    return moves;
}

// NANA_HASH_METER=0 (measurement): merge states differing only in pending garbage.
// Keep the meter in the hash by default.
#ifndef NANA_HASH_METER
#define NANA_HASH_METER 1
#endif

uint32_t EmulationGame::hash() const {
    Board board = game.board;
    int gheight = board.get_garbage_height();
    for (auto& row : board.board) {
        row = row >> gheight;
    }
    uint32_t hash = fasthash32(&board, sizeof(Board), 0x9012'3456);
    hash = fasthash32(&game.current_piece.type, sizeof(PieceType), hash);
    hash = fasthash32(&game.b2b, sizeof(u16), hash);
    hash = fasthash32(&game.combo, sizeof(u16), hash);
    if (game.hold)
        hash = fasthash32(&*game.hold, sizeof(*game.hold), hash);
#if NANA_HASH_QUEUE_FRONT
    else
        // with nothing held, holding plays the next piece: it decides the node's moves
        hash = fasthash32(&game.queue.front(), sizeof(PieceType), hash);
#endif
    if constexpr (NANA_HASH_METER) {
        for (int garbage : garbage_meter) {
            hash = fasthash32(&garbage, sizeof(garbage), hash);
        }
    }
    for (int garbage : phantom_meter) {
        hash = fasthash32(&garbage, sizeof(garbage), hash);
    }
    // garbage in transit is left out: no player sees it (at best they infer it)
    return hash;
}

float EmulationGame::app() {
    return (float) attack / (float) pieces;
}


float EmulationGame::true_app() {
    return (float)true_attack / (float)pieces;
}

float EmulationGame::apl() {
    return (float)attack / (float)lines;
}

float EmulationGame::b2b() {
    return (float) game.b2b; 
}

double EmulationGame::opponent_jeopardy() const {
    return markov.jeopardy();
}

double EmulationGame::opponent_deaths() const {
    return markov.deaths;
}

bool EmulationGame::opponent_dead() const {
    return markov.is_dead();
}

void EmulationGame::reset_opponent_rng() {
    markov.reset_rng();
}

void EmulationGame::reseed_chance() {
    if (deterministic_chance) {
        // Partial oracle: true pieces and opponent RNG, but sample unknown holes.
        if constexpr (NANA_ORACLE_RANDOM_HOLES) hole_rng = fast_seed();
        return;
    }
    rng.PPTRNG = fast_seed();
    if constexpr (NANA_SEPARATE_HOLES) hole_rng = fast_seed();
    const int left = rng.bagiterator;
    for (int i = (int)rng.bag.size() - 1; i > left; i--) {
        std::swap(rng.bag[i], rng.bag[left + rng.getRand(i - left + 1)]);
    }
    reset_opponent_rng();
}

void EmulationGame::start_from_root() {
    reseed_chance();
    markov.resample_from_root();
}

void EmulationGame::reset_opponent() {
    markov = MarkovOpponent();
}

void EmulationGame::reset_kill(float discount, bool expected) {
    kill_value = 0;
    kill_survival = 1;
    kill_weight = 1;
    kill_discount = discount;
    markov.expected_death = expected;
}

float EmulationGame::expected_kill() const {
    // death_value is in the same discount from the next piece on
    const float ahead = markov.death_value();
    return std::clamp(kill_value + kill_survival * kill_weight * ahead, 0.0f, 1.0f);
}

void EmulationGame::write_opponent_header(std::ostream& out) {
    out << "game,piece,sent,opp_attack,opp_sent,opp_meter,opp_dead\n";
}

void EmulationGame::write_opponent_step(std::ostream& out, uint64_t game_id) const {
    const double opp_attack = markov.last_attack;
    out << game_id << ',' << ramp_pieces << ',' << last_sent << ',' << opp_attack << ',' << last_received << ','
        << opponent_jeopardy() << ',' << (opponent_dead() ? 1 : 0) << '\n';
}
