#include "EmulationGame.hpp"
#include "engine/Utility.hpp"

#include "Util/fasthash.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>

EmulationGame::EmulationGame() {
    static const bool tetrio = [] {
        const char* rules = std::getenv("NANA_RULES");
        return rules && std::strcmp(rules, "tetrio") == 0;
    }();
    if (tetrio) {
        set_tetrio_rules();
    }
}

void EmulationGame::set_tetrio_rules() {
    game.mode = TetrioS1();
    stadium_rules = true;
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

    for (int& garbage : buffered_garbage) {
        garbage_meter.insert(garbage_meter.begin(), garbage);
    }

    buffered_garbage.clear();

    int garbage_amount = 0;

    // simulate opponent

    garbage_amount = markov_opponent ? (int)markov.play() : (int)opponent.play();

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

    const float death = markov_opponent ? (markov.expected_death ? markov.last_death_prob : (markov.dead ? 1.0f : 0.0f))
                                        : (opponent.dead ? 1.0f : 0.0f);
    kill_weight *= kill_discount;
    kill_value += kill_survival * death * kill_weight;
    kill_survival *= 1.0f - death;

    if (garbage_amount > 0) {
        if (stadium_rules) {
            // both players move at once: it meets our next piece's attack, and is tanked after it
            garbage_meter.insert(garbage_meter.begin(), garbage_amount);
        }
        else {
            buffered_garbage.insert(buffered_garbage.begin(), garbage_amount);
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

    spinType spin = move.piece.spin;
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

    // send damage
    if (markov_opponent) {
        markov.receiveAttack(damage);
    }
    else {
        opponent.receiveAttack(damage);
    }
    last_sent = damage;

    // combo block
    if (cleared_lines == 0 && stadium_rules) {
        // the whole meter at once, one hole
        int total = 0;
        for (int garbage : garbage_meter) {
            total += garbage;
        }
        if (total > 0) {
            game.add_garbage(total, rng.getRand(Board::width));
        }
        garbage_meter.clear();
    }
    else if (cleared_lines == 0) {

        int garbage_used = 0;

        // place garbage 
        while (garbage_meter.size() > 0) {
            int& incoming = garbage_meter.back();

            int garbage = std::min(incoming, 8 - garbage_used);

            game.add_garbage(garbage, rng.getRand(Board::width));

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

    //game.app = app();

};

void EmulationGame::objectively_play_moves() {


}


std::vector<Move> EmulationGame::legal_moves() {
    std::vector<Piece> raw_actions = game.get_possible_piece_placements();

    std::vector<Move> moves;

    for (auto& raw_action : raw_actions) {
        //moves.push_back(Move(raw_action, true));
        moves.push_back(Move(raw_action, false));
    }

    return moves;
}

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
    for (int garbage : garbage_meter) {
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
    return markov_opponent ? markov.jeopardy() : opponent.jeopardy();
}

double EmulationGame::opponent_deaths() const {
    return markov_opponent ? markov.deaths : opponent.deaths;
}

bool EmulationGame::opponent_dead() const {
    return markov_opponent ? markov.is_dead() : opponent.dead;
}

void EmulationGame::reset_opponent_rng() {
    if (markov_opponent) {
        markov.reset_rng();
    }
    else {
        opponent.reset_rng();
    }
}

void EmulationGame::start_from_root() {
    rng.PPTRNG = fast_seed();
    reset_opponent_rng();
    if (markov_opponent) {
        markov.resample_from_root();
    }
}

void EmulationGame::reset_opponent(bool markov) {
    markov_opponent = markov;
    if (markov) {
        this->markov = MarkovOpponent();
        // empty, so copying the state doesn't allocate
        opponent.stack.clear();
        opponent.garbage.clear();
    }
    else {
        opponent = Opponent();
        // Opponent() seeds itself from random_device and draws its queue from that: reseed from
        // fast_seed so a seeded search (fast_seed_reset) replays exactly
        opponent.rng.PPTRNG = fast_seed();
        opponent.nextI = opponent.rng.getRand(7);
        opponent.nextSpinPiece = opponent.rng.getRand(7);
    }
}

void EmulationGame::reset_kill(float discount, bool expected) {
    kill_value = 0;
    kill_survival = 1;
    kill_weight = 1;
    kill_discount = discount;
    markov.expected_death = expected && markov_opponent;
}

float EmulationGame::expected_kill() const {
    // death_value is in the same discount from the next piece on
    const float ahead = markov_opponent ? markov.death_value() : 0.0f;
    return std::clamp(kill_value + kill_survival * kill_weight * ahead, 0.0f, 1.0f);
}

void EmulationGame::write_opponent_header(std::ostream& out) {
    out << "game,piece,sent,opp_attack,opp_sent,opp_meter,opp_dead\n";
}

void EmulationGame::write_opponent_step(std::ostream& out, uint64_t game_id) const {
    const double opp_attack = markov_opponent ? markov.last_attack : opponent.last_attack;
    out << game_id << ',' << ramp_pieces << ',' << last_sent << ',' << opp_attack << ',' << last_received << ','
        << opponent_jeopardy() << ',' << (opponent_dead() ? 1 : 0) << '\n';
}
