// Positions and their scores under Eval::eval_CC_standard, to check the port against Cold Clear's own
// evaluator (cold-clear's eval_dump, which reads these lines). Plays seeded games greedily with the
// ported eval and, before every move, writes the position and the score of each placement of the
// current piece (no hold): "value" the board's score and "acc" the move's reward, in Cold Clear's units.
//
// usage: nana_eval_dump [--games K] [--seed S] [--pieces P]

#include "EmulateCommon.hpp"
#include "EmulationGame.hpp"
#include "Eval.hpp"
#include "Util/json.hpp"
#include "engine/Utility.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

static const char* pieceName(PieceType type) {
    switch (type) {
    case PieceType::S: return "S";
    case PieceType::Z: return "Z";
    case PieceType::J: return "J";
    case PieceType::L: return "L";
    case PieceType::T: return "T";
    case PieceType::O: return "O";
    case PieceType::I: return "I";
    default: return "?";
    }
}

static const char* rotationName(RotationDirection r) {
    switch (r) {
    case RotationDirection::North: return "north";
    case RotationDirection::East: return "east";
    case RotationDirection::South: return "south";
    default: return "west";
    }
}

int main(int argc, const char** argv) {
    int games = 5;
    uint64_t seed = 5000;
    int pieces = 50;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string arg = argv[i];
        if (arg == "--games") games = std::stoi(argv[i + 1]);
        else if (arg == "--seed") seed = std::stoull(argv[i + 1]);
        else if (arg == "--pieces") pieces = std::stoi(argv[i + 1]);
    }

    for (int g = 0; g < games; g++) {
        EmulationGame game = newGame(seed + g, 0, 1);
        for (int p = 0; p < pieces; p++) {
            if (Shaktris::Utility::collides(game.game.board, game.game.current_piece)) {
                break;
            }
            const std::vector<PieceType> bag_left(game.rng.bag.begin() + std::min<size_t>(game.rng.bagiterator, game.rng.bag.size()),
                                                  game.rng.bag.end());
            nlohmann::json position;
            nlohmann::json rows = nlohmann::json::array();
            for (int y = 0; y < 40; y++) {
                nlohmann::json row = nlohmann::json::array();
                for (int x = 0; x < 10; x++) {
                    row.push_back(y < (int)Board::height && game.game.board.get(x, y) ? 1 : 0);
                }
                rows.push_back(row);
            }
            position["board"] = rows;
            position["queue"] = nlohmann::json::array({ pieceName(game.game.current_piece.type) });
            for (PieceType q : game.game.queue) {
                position["queue"].push_back(pieceName(q));
            }
            position["hold"] = game.game.hold ? nlohmann::json(pieceName(*game.game.hold)) : nlohmann::json(nullptr);
            position["bag"] = nlohmann::json::array();
            for (PieceType b : bag_left) {
                position["bag"].push_back(pieceName(b));
            }
            position["b2b"] = (int)game.game.b2b;
            position["combo"] = (int)game.game.combo;

            nlohmann::json moves = nlohmann::json::array();
            const std::vector<Piece> placements = game.placements();
            double best = -1e18;
            int best_index = -1;
            for (size_t i = 0; i < placements.size(); i++) {
                const Piece& piece = placements[i];
                double move_part = 0;
                const double total = Eval::eval_CC_standard(game.game, Move(piece, false), bag_left, &move_part);
                if (total > best) {
                    best = total;
                    best_index = (int)i;
                }
                if (piece.type != game.game.current_piece.type) {
                    continue;
                }
                const long acc = std::lround(move_part * 10000);
                moves.push_back({
                    { "type", pieceName(piece.type) },
                    { "orientation", rotationName(piece.rotation) },
                    { "x", piece.position.x },
                    { "y", piece.position.y },
                    { "spin", piece.spin == spinType::normal ? "full" : piece.spin == spinType::mini ? "mini" : "none" },
                    { "value", std::lround(total * 10000) - 10000 - acc },
                    { "acc", acc },
                });
            }
            position["nana_moves"] = moves;
            std::cout << position.dump() << "\n";

            if (best_index < 0) {
                break;
            }
            game.set_move(Move(placements[best_index], false));
            game.play_moves();
            game.chance_move();
        }
    }
    return 0;
}
