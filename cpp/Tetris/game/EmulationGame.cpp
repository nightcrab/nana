#include "EmulationGame.hpp"
#include "Config.hpp"
#include "engine/RotationSystems.hpp"
#include "engine/Utility.hpp"

#include "Util/fasthash.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#ifdef NANA_SEARCH_FIXTURE
#include "tests/SearchFixture.hpp"
#endif

const char* EmulationGame::rules() {
    return config().rules.c_str();
}

namespace {
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
#ifdef NANA_SEARCH_FIXTURE
    if (fixture_active) return move;
#endif
    move.piece.position.y += game.board.get_garbage_height();
    return move;
};

template <class Opp>
void EmulationGame::chance_move() {
#ifdef NANA_SEARCH_FIXTURE
    if (fixture_active) { search_fixture::advance(*this); return; }
#endif
    for (auto& piece : game.queue) {
        if (piece == PieceType::Empty) {
            piece = rng.getPiece();
        }
    }

    const bool tetrio = std::holds_alternative<TetrioS1>(game.mode);

    if (garbage_delay > 0) {
        buffered_garbage.resize(garbage_delay);
        if (buffered_garbage.front() > 0) {
            garbage_meter.insert(garbage_meter.begin(), buffered_garbage.front());
        }
        std::shift_left(buffered_garbage.begin(), buffered_garbage.end(), 1);
        buffered_garbage.back() = 0;
    }

    // garbage already on its way still lands; without an opponent nothing new is sent
    if constexpr (!Opp::simulated) {
        last_received = 0;
        return;
    }

    int garbage_amount = (int)markov.play();
    if (phantom_garbage && garbage_amount > 0) {
        phantom_meter.insert(phantom_meter.begin(), garbage_amount);
    }

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

    if (garbage_amount > 0) {
        if (garbage_delay == 0) {
            garbage_meter.insert(garbage_meter.begin(), garbage_amount);
        }
        else {
            buffered_garbage.back() = garbage_amount;
        }
    }
};

template void EmulationGame::chance_move<MarkovOpponent>();
template void EmulationGame::chance_move<NoOpponent>();

template <class Opp>
void EmulationGame::play_moves(){
#ifdef NANA_SEARCH_FIXTURE
    if (fixture_active) {
        fixture_action = move.piece.position.x;
        ++pieces;
        return;
    }
#endif
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

    if (Opp::simulated && phantom_garbage) {
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

    if (tetrio) {
        for (int& incoming : buffered_garbage) {
            const int cancelled = std::min(damage, incoming);
            incoming -= cancelled;
            damage -= cancelled;
        }
    }

    // send damage
    if constexpr (Opp::simulated) {
        if (tetrio && garbage_delay > 0) {
            outgoing_garbage.resize(garbage_delay);
            outgoing_garbage.back() += damage;
        }
        else {
            markov.receiveAttack(damage);
        }
    }
    last_sent = damage;

    // combo block
    if (cleared_lines == 0 && tetrio) {
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

    if (Opp::simulated && track_belief && tetrio) {
        belief.turn(sent, cleared_lines > 0, landed);
    }

    //game.app = app();

};

template void EmulationGame::play_moves<MarkovOpponent>();
template void EmulationGame::play_moves<NoOpponent>();

void EmulationGame::objectively_play_moves() {


}


namespace {
using SpinTable = std::array<std::array<std::array<uint8_t, Board::height>, Board::width>, 4>;

void t_spins(const Board& board, SpinTable& best) {
    const Piece spawn(PieceType::T);
    if (Shaktris::Utility::collides(board, spawn)) {
        return;
    }
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
        return pieces;
    }

    // eval "convex" plays no spins, so its placements carry no spin labels
    static const bool label_spins = evalMode() != EvalMode::Convex;
    bool candidates = false;
    for (Piece& piece : pieces) {
        piece.spin = spinType::null;
        candidates |= label_spins && piece.type == PieceType::T && spin_corners(game.board, piece);
    }
    if (!candidates) {
        return pieces;
    }

    SpinTable best{};
    t_spins(game.board, best);
    for (Piece& piece : pieces) {
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

#ifndef NANA_HASH_METER
#define NANA_HASH_METER 1
#endif

uint32_t EmulationGame::hash() const {
#ifdef NANA_SEARCH_FIXTURE
    if (fixture_active) return search_fixture::hash_base + fixture_node;
#endif
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
    else
        hash = fasthash32(&game.queue.front(), sizeof(PieceType), hash);
    if constexpr (NANA_HASH_METER) {
        for (int garbage : garbage_meter) {
            hash = fasthash32(&garbage, sizeof(garbage), hash);
        }
    }
    for (int garbage : phantom_meter) {
        hash = fasthash32(&garbage, sizeof(garbage), hash);
    }
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

template <class Opp>
void EmulationGame::reseed_chance() {
    if (deterministic_chance) {
        if constexpr (NANA_ORACLE_RANDOM_HOLES) hole_rng = fast_seed();
        return;
    }
    rng.PPTRNG = fast_seed();
    if constexpr (NANA_SEPARATE_HOLES) hole_rng = fast_seed();
    const int left = rng.bagiterator;
    for (int i = (int)rng.bag.size() - 1; i > left; i--) {
        std::swap(rng.bag[i], rng.bag[left + rng.getRand(i - left + 1)]);
    }
    if constexpr (Opp::simulated) {
        reset_opponent_rng();
    }
}

template void EmulationGame::reseed_chance<MarkovOpponent>();
template void EmulationGame::reseed_chance<NoOpponent>();

template <class Opp>
void EmulationGame::start_from_root() {
    reseed_chance<Opp>();
    if constexpr (Opp::simulated) {
        markov.resample_from_root();
    }
}

template void EmulationGame::start_from_root<MarkovOpponent>();
template void EmulationGame::start_from_root<NoOpponent>();

void EmulationGame::reset_opponent() {
    markov = MarkovOpponent();
}

void EmulationGame::write_opponent_header(std::ostream& out) {
    out << "game,piece,sent,opp_attack,opp_sent,opp_meter,opp_dead\n";
}

void EmulationGame::write_opponent_step(std::ostream& out, uint64_t game_id) const {
    const double opp_attack = markov.last_attack;
    out << game_id << ',' << ramp_pieces << ',' << last_sent << ',' << opp_attack << ',' << last_received << ','
        << opponent_jeopardy() << ',' << (opponent_dead() ? 1 : 0) << '\n';
}
