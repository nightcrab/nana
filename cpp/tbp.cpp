#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

#include "Search.hpp"
#include "Config.hpp"
#include "OpponentEncoder.hpp"
#include "OpponentFilter.hpp"
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
auto json_to_type(nlohmann::json json) -> PieceType;

static std::vector<PieceType> json_to_bag(const nlohmann::json& message) {
    std::vector<PieceType> left;
    if (!message.contains("randomizer") || message["randomizer"].value("type", "") != "seven_bag") {
        return left;
    }
    for (const auto& piece : message["randomizer"]["bag_state"]) {
        const PieceType type = json_to_type(piece);
        if (type == PieceType::Empty || std::find(left.begin(), left.end(), type) != left.end()) {
            return {};
        }
        left.push_back(type);
    }
    return left;
}

static void apply_bag(RNG& rng, const std::vector<PieceType>& left) {
    rng.bagiterator = uint8_t(rng.bag.size() - left.size());
    std::copy(left.begin(), left.end(), rng.bag.begin() + rng.bagiterator);
}

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

struct SeenOpponent {
    bool have = false;
    Board board{};
    int combo = 0;
    int b2b = 0;
};

static bool readOpponent(const nlohmann::json& player, SeenOpponent& out) {
    try {
        if (!player.is_object() || !player.contains("board")) {
            return false;
        }
        out.board = json_to_board(player["board"]);
        out.combo = player.contains("combo") && player["combo"].is_number() ? player["combo"].get<int>() : 0;
        out.b2b = 0;
        if (player.contains("back_to_back")) {
            const auto& v = player["back_to_back"];
            out.b2b = v.is_boolean() ? (v.get<bool>() ? 1 : 0) : (v.is_number() ? v.get<int>() : 0);
        }
        out.have = true;
        return true;
    }
    catch (const std::exception&) {
        return false;
    }
}

static RootOpponent rootOpponent(const OpponentFilter& filter, const SeenOpponent& seen, OpponentFilter& search) {
    std::array<float, markov::STATES> state;
    std::array<float, markov::SINCE> since;
    search = filter;
    if (!seen.have) {
        return filter.root();
    }
    bool empty = true;
    for (auto column : seen.board.board) {
        empty = empty && column == 0;
    }
    if (!filter.started()) {
        RootOpponent out;
        out.fresh = empty || !markov::encodeOpponent(seen.board, seen.combo, seen.b2b, out.state, out.since);
        search = OpponentFilter::fromRoot(out);
        return out;
    }
    if (markov::opponentEvidence(seen.board, seen.combo, seen.b2b, state, since)) {
        search.applyEvidence(&state, &since);
        return filter.root(&state, &since);
    }
    return filter.root();
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
        loadTbpConfig();
        int core_count = config().threads;

        // send info state to the client
        std::vector<PieceType> queue;

        // generated pieces that we know about
        int total_pieces = 0;
        std::vector<PieceType> bag_left;

        EmulationGame game;

        OpponentFilter opponent_filter;
        SeenOpponent seen_opponent;
        RootOpponent root_opponent;
        OpponentFilter search_filter;
        auto refreshRoot = [&] {
            root_opponent = rootOpponent(opponent_filter, seen_opponent, search_filter);
            Search::root_opponent = &root_opponent;
            Search::root_filter = &search_filter;
        };
        bool have_suggestion = false;
        Move last_suggestion;

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
        message["features"] = nlohmann::json::array({ "set_threads", "set_thinks" });

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
                if (Search::searching) {
                    Search::endSearch();
                }
                break;
            }
            else if (type == "rules") {
                if (message.contains("threads") && !Search::initialised) {
                    int threads;
                    CHECK_ERR(threads = message["threads"].get<int>());
                    if (threads >= 1) {
                        core_count = threads;
                    }
                }

                if (message.contains("thinks")) {
                    uint64_t thinks;
                    CHECK_ERR(thinks = message["thinks"].get<uint64_t>());
                    Search::simulation_limit = thinks;
                }

                if (message.contains("garbage_delay")) {
                    int delay;
                    CHECK_ERR(delay = message["garbage_delay"].get<int>());
                    if (delay >= 0) {
                        EmulationGame::set_garbage_delay(delay);
                    }
                }

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
                CHECK_ERR(back_to_back = player_1["back_to_back"].is_boolean() ? (player_1["back_to_back"].get<bool>() ? 1 : 0)
                                                                              : player_1["back_to_back"].get<int>());

                int combo;
                CHECK_ERR(combo = player_1["combo"].get<int>());

                if (have_suggestion) {
                    EmulationGame expected = game;
                    const int attack_before = expected.attack;
                    const int lines_before = expected.lines;
                    expected.set_move(last_suggestion);
                    expected.play_moves();
                    const int landed = garbageRows(expected.game.board, board);
                    if (landed >= 0) {
                        const bool cleared = expected.lines > lines_before && landed == 0;
                        opponent_filter.turn(expected.attack - attack_before, cleared, landed);
                    }
                    else {
                        opponent_filter.reset();
                    }
                }
                else {
                    opponent_filter.reset();
                }
                have_suggestion = false;

                // create new emulation game rooted at this state
                game = EmulationGame();
#ifdef TBP1
#else
                {
                    // create opponent model from other board
                    nlohmann::json player_2;

                    // tbp 2
                    CHECK_ERR(player_2 = message["opponents"][0]);
                    readOpponent(player_2, seen_opponent);
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

                bag_left = json_to_bag(message);
                if (!bag_left.empty()) {
                    apply_bag(game.rng, bag_left);
                }

                // end previous search this can happen if damage was sent and immediately after the game is over
                if (Search::searching) {
                    Search::endSearch();
                }

                refreshRoot();
                if (!Search::initialised) {
                    Search::startSearch(game, core_count);
                }
                else {
                    Search::continueSearch(game);
                }
            }
            else if (type == "new_piece") {
                PieceType piece = json_to_type(message["piece"]);
                queue.push_back(piece);

                total_pieces += 1;

                const auto dealt = std::find(bag_left.begin(), bag_left.end(), piece);
                if (dealt != bag_left.end()) {
                    bag_left.erase(dealt);
                    if (bag_left.empty()) {
                        bag_left = { PieceType::S, PieceType::Z, PieceType::J, PieceType::L,
                                     PieceType::T, PieceType::O, PieceType::I };
                    }
                    apply_bag(game.rng, bag_left);
                }
                else {
                    bag_left.clear();
                    game.rng.bagiterator = total_pieces % 7;
                }
            }
            else if (type == "suggest") {
                if (Search::simulation_limit > 0 && Search::searching) {
                    Search::waitForSimulations(std::chrono::seconds(120));
                }
                Search::endSearch();
                Move best_move = game.specific_move(Search::bestMove());
                last_suggestion = best_move;
                have_suggestion = !best_move.null_move;

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
                const nlohmann::json full = message;
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
                    if (full.contains("opponents") && full["opponents"].is_array() && !full["opponents"].empty()) {
                        player_2 = full["opponents"][0];
                        readOpponent(player_2, seen_opponent);
                    }
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

                const int attack_before = game.attack;
                const int lines_before = game.lines;
                game.play_moves();

                opponent_filter.turn(game.attack - attack_before, game.lines > lines_before, 0);
                have_suggestion = false;

                refreshRoot();
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