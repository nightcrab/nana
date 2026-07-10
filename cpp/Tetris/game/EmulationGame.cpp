#include "EmulationGame.hpp"
#include "engine/Utility.hpp"

#include "Util/fasthash.h"
#include <iostream>

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

    garbage_amount = opponent.play();

    if (garbage_amount > 0) {
        buffered_garbage.insert(buffered_garbage.begin(), garbage_amount);
    }
};

void EmulationGame::play_moves(){
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
    opponent.receiveAttack(damage);

    // combo block
    if (cleared_lines == 0) {

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
    for (int garbage : garbage_meter) {
        hash = fasthash32(&garbage, sizeof(garbage), hash);
    }
    return hash;
}

float EmulationGame::app() const {
    return (float) attack / (float) pieces;
}


float EmulationGame::true_app() const {
    return (float)true_attack / (float)pieces;
}

float EmulationGame::apl() {
    return (float)attack / (float)lines;
}

float EmulationGame::b2b() {
    return (float) game.b2b; 
}