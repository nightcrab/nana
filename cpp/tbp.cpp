#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

#include "Search.hpp"
#include "Util/json.hpp"

auto type_to_json(PieceType type) -> nlohmann::json {
    switch (type) {
    case PieceType::S:
        return "S";
        break;
    case PieceType::Z:
        return "Z";
        break;
    case PieceType::J:
        return "J";
        break;
    case PieceType::L:
        return "L";
        break;
    case PieceType::T:
        return "T";
        break;
    case PieceType::O:
        return "O";
        break;
    case PieceType::I:
        return "I";
        break;
    case PieceType::Empty:
        return "I";
        break;
    default:
        return nullptr;
        break;
    }
};
auto json_to_type(nlohmann::json json) -> PieceType {

    if (json.is_null()) {
        return PieceType::Empty;
    }

    std::string str = json.get<std::string>();

    if (str == "S") {
        return PieceType::S;
    }
    else if (str == "Z") {
        return PieceType::Z;
    }
    else if (str == "J") {
        return PieceType::J;
    }
    else if (str == "L") {
        return PieceType::L;
    }
    else if (str == "T") {
        return PieceType::T;
    }
    else if (str == "O") {
        return PieceType::O;
    }
    else if (str == "I") {
        return PieceType::I;
    }
    else {
        return PieceType::Empty;
    }
};
auto board_to_json(Board board) -> nlohmann::json {
    nlohmann::json json;
    std::array<std::array<std::optional<std::string>, 40>, 10> tbpBoard;

    for (int x = 0; x < Board::width; ++x)
        for (int y = 0; y < Board::visual_height; ++y) {
            if (board.get(x, y))
                tbpBoard[x][y] = "G";
            else
                tbpBoard[x][y] = std::nullopt;
        }

    json = nlohmann::json::array();

    for (int y = 0; y < Board::height; ++y) {
        nlohmann::json tmp = nlohmann::json::array();
        for (int x = 0; x < Board::width; ++x) {
            if (tbpBoard[x][y].has_value()) {
                tmp.push_back(tbpBoard[x][y].value());
            }
            else {
                tmp.push_back(nullptr);
            }
        }
        json.push_back(tmp);
    }
    return json;
};
auto json_to_board(nlohmann::json json) -> Board {
    Board board;
    for (int y = 0; y < Board::height; ++y) {
        for (int x = 0; x < Board::width; ++x) {
            if (!json[y][x].is_null()) {
                board.set(x, y);
            }
        }
    }
    return board;
};
auto json_to_queue(nlohmann::json json) -> std::vector<PieceType> {
    std::vector<PieceType> queue;
    for (auto& piece : json) {
        queue.push_back(json_to_type(piece));
    }
    return queue;
};
auto rotation_to_json(RotationDirection dir) -> nlohmann::json {
    if (dir == RotationDirection::North) {
        return "north";
    }
    else if (dir == RotationDirection::East) {
        return "east";
    }
    else if (dir == RotationDirection::South) {
        return "south";
    }
    else if (dir == RotationDirection::West) {
        return "west";
    }
    throw;
    return nullptr;
}
auto json_to_rotation(nlohmann::json json) -> RotationDirection {
    std::string str = json.get<std::string>();
    if (str == "north") {
        return RotationDirection::North;
    }
    else if (str == "east") {
        return RotationDirection::East;
    }
    else if (str == "south") {
        return RotationDirection::South;
    }
    else if (str == "west") {
        return RotationDirection::West;
    }
    throw std::runtime_error("invalid rotation");
    return RotationDirection::North;
}
auto spin_to_json(spinType spin) -> nlohmann::json {
    if (spin == spinType::null) {
        return "none";
    }
    else if (spin == spinType::mini) {
        return "mini";
    }
    else if (spin == spinType::null) {
        return "none";
    }
    else if (spin == spinType::normal) {
        return "full";
    }
    throw;
    return nullptr;
}
auto move_to_json(Move& move) {
    nlohmann::json json;
    json["location"]["x"] = move.piece.position.x;
    json["location"]["y"] = move.piece.position.y;
    json["location"]["type"] = type_to_json(move.piece.type);
    json["location"]["orientation"] = rotation_to_json(move.piece.rotation);
    json["spin"] = spin_to_json(move.piece.spin);
    return json;
}
auto json_to_spin(nlohmann::json json) -> spinType {
    std::string str = json.get<std::string>();
    if (str == "none") {
        return spinType::null;
    }
    else if (str == "mini") {
        return spinType::mini;
    }
    else if (str == "full") {
        return spinType::normal;
    }
    throw std::runtime_error("invalid spin");
    return spinType::null;
}

#define STR2(x) #x
#define STR(x) STR2(x)
#define CHECK_ERR(code) try { code; } catch (std::exception &e) { std::cerr << __LINE__ << ": " << STR(code) << std::endl << e.what() << std::endl; return 1; }

// TBP entry point
int main() {
    try {
        // search threads; NANA_THREADS overrides, e.g. to run several bots on one machine
        int CORE_COUNT = 3;
        if (const char* text = std::getenv("NANA_THREADS")) {
            CORE_COUNT = std::max(1, std::atoi(text));
        }

        // send info state to the client
        std::vector<PieceType> queue;

        // generated pieces that we know about
        int total_pieces = 0;

        EmulationGame game;

        nlohmann::json message;

        message["type"] = "info";
        /*
        name 	A string, the human-friendly name of the bot. Example: "Cold Clear"
        version 	A string version identifier. Example: "Gen 14 proto 8"
        author 	A string identifying the author of the bot. Example: "SoRA_X7"
        features 	A list of supported features.*/

        message["name"] = "Nana";
        message["version"] = "0.1";
        message["author"] = "Shakkar + Awyzza";
        // part of the TBP info message; clients like the stadium require it
        message["features"] = nlohmann::json::array();

        // send just output to stdout
        std::cout << message.dump() << std::endl;

        std::string str_input;

        while (true) {
            str_input.clear();
            message.clear();

            std::cin >> str_input;
            message = nlohmann::json::parse(str_input);
            std::string type = message["type"];

            if (type == "quit") {
                Search::endSearch();
                break;
            }
            else if (type == "rules") {

                // currently there really is no use for this because both the client and spec doesnt really use this

                // send the ready message to the client
                message.clear();
                message["type"] = "ready";
                std::cout << message.dump() << std::endl;
            }
            else if (type == "start") {
                /*
                The `start` message tells the bot to begin calculating from the specified
                position. This message must be sent before asking the bot for a move. This
                message must be an array of the specified attributes, each element in the
                array being a player. The player the bot will be acting on should be on index
                0.


                Attribute       | Description
                ---------       | -----------
                `hold`          | Either a pieces if hold is filled or `null` if hold is empty.
                `queue`         | A list of pieces. Example: `["S", "Z", "O"]`
                `combo`         | The number of consecutive line clears that have been made.
                `back_to_back`  | An integer indicating back to back status.
                `board`         | A list of 40 lists of 10 board cells..*/

                nlohmann::json player_1;
                CHECK_ERR(player_1 = message);

                Board board;

                CHECK_ERR(board = json_to_board(player_1["board"]));


                CHECK_ERR(queue = json_to_queue(player_1["queue"]));

                PieceType hold;

                CHECK_ERR(hold = json_to_type(player_1["hold"]));

                int back_to_back;
                // TBP sends a bool; some clients (the stadium) send the b2b count
                CHECK_ERR(back_to_back = player_1["back_to_back"].is_boolean() ? (player_1["back_to_back"].get<bool>() ? 1 : 0)
                                                                              : player_1["back_to_back"].get<int>());

                int combo;
                CHECK_ERR(combo = player_1["combo"].get<int>());

                // create new emulation game rooted at this state
                game = EmulationGame();
#ifdef TBP1
#else
                {
                    // create opponent model from other board
                    nlohmann::json player_2;

                    // tbp 2
                    CHECK_ERR(player_2 = message["opponents"][0]);
                    Board board2;

                    CHECK_ERR(board2 = json_to_board(player_2["board"]));
                    // there is more information that isnt just the board, such as hold, queue, combo, b2b
                    // but the opponent constructor only needs the board so thats all im going to pull

                    Game game2;
                    game2.board = board2;
                    game.opponent = Opponent(game2);
                }
#endif

                game.game.board = board;
                game.game.current_piece = queue.front();
                queue.erase(queue.begin());

                for (int i = 0; i < game.game.queue.size(); ++i) {
                    game.game.queue[i] = PieceType::Empty;
                }

                for (int i = 0; i < game.game.queue.size(); ++i) {
                    if (game.game.queue[i] == PieceType::Empty) {
                        if (queue.size() > 0) {
                            game.game.queue[i] = queue.front();
                            queue.erase(queue.begin());
                        }
                    }
                }

                if (total_pieces == 0) {
                    total_pieces = queue.size();
                }

                if (hold == PieceType::Empty) {
                    game.game.hold = std::nullopt;
                }
                else {
                    game.game.hold = hold;
                }
                game.game.combo = combo;
                game.game.b2b = back_to_back;

                // end previous search this can happen if damage was sent and immediately after the game is over
                if (Search::searching) {
                    Search::endSearch();
                }

                if (!Search::initialised) {
                    Search::startSearch(game, CORE_COUNT);
                }
                else {
                    Search::continueSearch(game);
                }
            }
            else if (type == "new_piece") {
                PieceType piece = json_to_type(message["piece"]);
                queue.push_back(piece);

                total_pieces += 1;

                game.rng.bagiterator = total_pieces % 7;
            }
            else if (type == "suggest") {
                // end search and give the best move
                Search::endSearch();
                Move best_move = game.specific_move(Search::bestMove());

                // create suggestion message
                message.clear();
                message["type"] = "suggestion";
                message["moves"] = nlohmann::json::array();
                message["moves"][0] = move_to_json(best_move);
                message["moves"][0]["inputs"] = nlohmann::json::array();

                // TODO: add inputs

                std::cout << message.dump() << std::endl;
            }
            else if (type == "play") {
                // play the move
                message = message["move"];
                Move move;
                move.null_move = false;
                CHECK_ERR(move.piece = json_to_type(message["location"]["type"]));
                CHECK_ERR(move.piece.position.x = message["location"]["x"].get<int>());
                CHECK_ERR(move.piece.position.y = message["location"]["y"].get<int>());
                CHECK_ERR(move.piece.spin = json_to_spin(message["spin"]));

                RotationDirection rotation;
                CHECK_ERR(rotation = json_to_rotation(message["location"]["orientation"]));


#ifdef TBP1
#else
                {
                    // create opponent model from other board
                    nlohmann::json player_2;

                    // tbp 2
                    CHECK_ERR(player_2 = message["opponents"][0]);
                    Board board2;

                    CHECK_ERR(board2 = json_to_board(player_2["board"]));
                    // there is more information that isnt just the board, such as hold, queue, combo, b2b
                    // but the opponent constructor only needs the board so thats all im going to pull

                    Game game2;
                    game2.board = board2;
                    game.opponent = Opponent(game2);
                }
#endif



                for(int i = 0; i < int(rotation); ++i) {
					move.piece.calculate_rotate(TurnDirection::Right);
				}

                // play the move
                game.set_move(move);


                for (int i = 0; i < game.game.queue.size(); ++i) {
                    if (game.game.queue[i] == PieceType::Empty) {
                        game.game.queue[i] = queue.front();
                        queue.erase(queue.begin());
                    }
                }

                game.play_moves();
                
                Search::continueSearch(game);
            }
        }
    }
    catch (const std::exception &e) {
        std::cout << "exception thrown: " << e.what() << std::endl;
        std::cerr << "exception thrown: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}