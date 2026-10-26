// dumps positions scored by Eval::evalCC, to check the port against Cold Clear
// usage: nana_eval_dump [--games K] [--seed S] [--pieces P] [--terms 1]
// --terms 1: instead, one line per position with each candidate move's weighted cc terms (every move, hold included)
// --positions FILE: with --terms 1, score the positions in FILE (JSON lines with board: 200 cells, row 0 at the
//                   bottom; current, hold, queue, bag_left as piece letters; b2b; combo) instead of greedy games
// --time R: with --positions, time each subset of Eval::evalCCTimed over every candidate move, best of R passes

#include "EmulateCommon.hpp"
#include "EmulationGame.hpp"
#include "Eval.hpp"
#include "Util/json.hpp"
#include "engine/Utility.hpp"

#include <array>
#include <chrono>
#include <fstream>
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
    bool terms_mode = false;
    std::string positions_file;
    int time_reps = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string arg = argv[i];
        if (arg == "--games") games = std::stoi(argv[i + 1]);
        else if (arg == "--seed") seed = std::stoull(argv[i + 1]);
        else if (arg == "--pieces") pieces = std::stoi(argv[i + 1]);
        else if (arg == "--terms") terms_mode = std::stoi(argv[i + 1]) != 0;
        else if (arg == "--positions") positions_file = argv[i + 1];
        else if (arg == "--time") time_reps = std::stoi(argv[i + 1]);
    }
    if (terms_mode && time_reps == 0) {
        std::cout << nlohmann::json({ { "names", Eval::ccTermNames } }).dump() << "\n";
    }
    struct Scored {
        Game game;
        std::vector<PieceType> bag_left;
        std::vector<Piece> placements;
    };
    std::vector<Scored> scored;
    if (terms_mode && !positions_file.empty()) {
        std::ifstream in(positions_file);
        if (!in) {
            std::cerr << "can't read " << positions_file << std::endl;
            return 1;
        }
        auto piece = [](const nlohmann::json& j) {
            if (j.is_null()) return PieceType::Empty;
            const std::string name = j.get<std::string>();
            for (PieceType t : { PieceType::S, PieceType::Z, PieceType::J, PieceType::L, PieceType::T, PieceType::O, PieceType::I }) {
                if (name == pieceName(t)) return t;
            }
            return PieceType::Empty;
        };
        std::string line;
        while (std::getline(in, line)) {
            const nlohmann::json pos = nlohmann::json::parse(line);
            EmulationGame game = newGame(0, 0, 1);
            game.game.board = Board{};
            for (int i = 0; i < 200; i++) {
                if (pos["board"][i].get<int>()) game.game.board.set(i % 10, i / 10);
            }
            game.game.current_piece = Piece(piece(pos["current"]));
            game.game.hold = pos["hold"].is_null() ? std::nullopt : std::optional<PieceType>(piece(pos["hold"]));
            game.game.queue.fill(PieceType::Empty);
            for (size_t q = 0; q < pos["queue"].size() && q < game.game.queue.size(); q++) {
                game.game.queue[q] = piece(pos["queue"][q]);
            }
            game.game.b2b = (u16)pos["b2b"].get<int>();
            game.game.combo = (u16)pos["combo"].get<int>();
            std::vector<PieceType> bag_left;
            for (const auto& b : pos["bag_left"]) bag_left.push_back(piece(b));
            const std::vector<Piece> placements = game.placements();
            if (time_reps > 0) {
                scored.push_back({ game.game, bag_left, placements });
                continue;
            }
            nlohmann::json totals = nlohmann::json::array(), terms = nlohmann::json::array();
            for (const Piece& placement : placements) {
                std::array<double, Eval::CC_TERMS> t;
                const double total = Eval::evalCCTerms(game.game, placement, bag_left, t);
                double sum = 0;
                for (double v : t) sum += v;
                if (std::abs(sum - (total - 1) * 10000) > 1e-6 * std::max(1.0, std::abs(sum))) {
                    std::cerr << "terms don't add up: " << sum << " vs " << (total - 1) * 10000 << std::endl;
                    return 1;
                }
                totals.push_back(sum);
                terms.push_back(t);
            }
            std::cout << nlohmann::json({ { "id", pos["id"] }, { "total", totals }, { "terms", terms } }).dump() << "\n";
        }
        if (time_reps > 0) {
            size_t evals = 0;
            for (const Scored& s : scored) evals += s.placements.size();
            std::cout << scored.size() << " positions, " << evals << " evals per pass, best of " << time_reps
                      << " passes (subsets interleaved per pass)\n";
            const int n = Eval::ccTimedCount();
            std::vector<double> best(n, 1e300);
            double sink = 0;
            for (int r = 0; r < time_reps; r++) {
                for (int i = 0; i < n; i++) {
                    const auto start = std::chrono::steady_clock::now();
                    for (const Scored& s : scored) {
                        for (const Piece& placement : s.placements) {
                            sink += Eval::evalCCTimed(i, s.game, placement, s.bag_left);
                        }
                    }
                    const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
                    best[i] = std::min(best[i], ns / evals);
                }
            }
            const int sets = Eval::ccTimedSets();
            const int base = sets - 1;
            for (int i = 0; i < n; i++) {
                std::cout << (i < sets ? "set    " : "alone  ") << Eval::ccTimedName(i) << ": " << best[i] << " ns/eval";
                if (i >= sets) std::cout << "  (+" << best[i] - best[base] << " over base)";
                std::cout << "\n";
            }
            std::cerr << "checksum " << sink << std::endl;
        }
        return 0;
    }

    for (int g = 0; g < games; g++) {
        EmulationGame game = newGame(seed + g, 0, 1);
        for (int p = 0; p < pieces; p++) {
            if (Shaktris::Utility::collides(game.game.board, game.game.current_piece)) {
                break;
            }
            const std::vector<PieceType> bag_left(game.rng.bag.begin() + std::min<size_t>(game.rng.bagiterator, game.rng.bag.size()),
                                                  game.rng.bag.end());
            if (terms_mode) {
                const std::vector<Piece> placements = game.placements();
                nlohmann::json totals = nlohmann::json::array(), terms = nlohmann::json::array();
                double best = -1e18;
                int best_index = -1;
                for (size_t i = 0; i < placements.size(); i++) {
                    std::array<double, Eval::CC_TERMS> t;
                    const double total = Eval::evalCCTerms(game.game, placements[i], bag_left, t);
                    double sum = 0;
                    for (double v : t) sum += v;
                    if (std::abs(sum - (total - 1) * 10000) > 1e-6 * std::max(1.0, std::abs(sum))) {
                        std::cerr << "terms don't add up: " << sum << " vs " << (total - 1) * 10000 << std::endl;
                        return 1;
                    }
                    totals.push_back(sum);
                    terms.push_back(t);
                    if (total > best) {
                        best = total;
                        best_index = (int)i;
                    }
                }
                std::cout << nlohmann::json({ { "game", g }, { "piece", p }, { "total", totals }, { "terms", terms } }).dump() << "\n";
                if (best_index < 0) {
                    break;
                }
                game.set_move(Move(placements[best_index], false));
                game.play_moves();
                game.chance_move();
                continue;
            }
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
                const double total = Eval::evalCC(game.game, piece, bag_left, &move_part);
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
