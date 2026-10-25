// survival benchmark for any TBP bot, on nana_emulate's seeded games
// usage: nana_emulate_tbp --bot PATH [--thinks N | --ms T] [--games K] [--seed S] [--max-pieces P]
//                         [--threads N] [--cpu K] [--ramp R] [--garbage-scale X]
//                         [--no-bag 1]  (start without the randomizer's bag state)

#include "EmulateCommon.hpp"
#include "EmulationGame.hpp"
#include "Util/Affinity.hpp"
#include "Util/json.hpp"
#include "engine/Utility.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif

class BotProcess {
   public:
    void start(const std::string& path) {
#ifdef _WIN32
        SECURITY_ATTRIBUTES sa{ sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE };
        HANDLE out_write, in_read;
        if (!CreatePipe(&out_read, &out_write, &sa, 0) || !CreatePipe(&in_read, &in_write, &sa, 0)) {
            throw std::runtime_error("can't make pipes");
        }
        SetHandleInformation(out_read, HANDLE_FLAG_INHERIT, 0);
        SetHandleInformation(in_write, HANDLE_FLAG_INHERIT, 0);
        STARTUPINFOA startup{};
        startup.cb = sizeof(startup);
        startup.hStdOutput = out_write;
        startup.hStdInput = in_read;
        startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
        startup.dwFlags |= STARTF_USESTDHANDLES;
        std::string command = "\"" + path + "\"";
        if (!CreateProcessA(nullptr, command.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &startup, &process)) {
            throw std::runtime_error("can't start " + path);
        }
        CloseHandle(out_write);
        CloseHandle(in_read);
#else
        int to_child[2], from_child[2];
        if (pipe(to_child) == -1 || pipe(from_child) == -1) {
            throw std::runtime_error("can't make pipes");
        }
        pid = fork();
        if (pid == -1) {
            throw std::runtime_error("can't fork");
        }
        if (pid == 0) {
            dup2(to_child[0], STDIN_FILENO);
            dup2(from_child[1], STDOUT_FILENO);
            close(to_child[0]);
            close(to_child[1]);
            close(from_child[0]);
            close(from_child[1]);
            execl(path.c_str(), path.c_str(), (char*)nullptr);
            perror("execl");
            _exit(127);
        }
        close(to_child[0]);
        close(from_child[1]);
        write_fd = to_child[1];
        read_fd = from_child[0];
        signal(SIGPIPE, SIG_IGN);
#endif
    }

    void send(const std::string& line) {
        const std::string data = line + "\n";
        size_t done = 0;
        while (done < data.size()) {
#ifdef _WIN32
            DWORD wrote = 0;
            if (!WriteFile(in_write, data.data() + done, (DWORD)(data.size() - done), &wrote, nullptr)) {
                throw std::runtime_error("the bot stopped reading");
            }
#else
            const ssize_t wrote = write(write_fd, data.data() + done, data.size() - done);
            if (wrote <= 0) {
                throw std::runtime_error("the bot stopped reading");
            }
#endif
            done += (size_t)wrote;
        }
    }

    std::string receive() {
        while (true) {
            const size_t end = pending.find('\n');
            if (end != std::string::npos) {
                std::string line = pending.substr(0, end);
                pending.erase(0, end + 1);
                if (!line.empty() && line.back() == '\r') {
                    line.pop_back();
                }
                if (!line.empty()) {
                    return line;
                }
                continue;
            }
            char buffer[4096];
#ifdef _WIN32
            DWORD got = 0;
            if (!ReadFile(out_read, buffer, sizeof(buffer), &got, nullptr) || got == 0) {
                throw std::runtime_error("the bot closed its output");
            }
#else
            const ssize_t got = read(read_fd, buffer, sizeof(buffer));
            if (got <= 0) {
                throw std::runtime_error("the bot closed its output");
            }
#endif
            pending.append(buffer, (size_t)got);
        }
    }

    void finish() {
#ifdef _WIN32
        CloseHandle(in_write);
        WaitForSingleObject(process.hProcess, 10000);
        CloseHandle(out_read);
        CloseHandle(process.hProcess);
        CloseHandle(process.hThread);
#else
        close(write_fd);
        int status = 0;
        waitpid(pid, &status, 0);
        close(read_fd);
#endif
    }

   private:
#ifdef _WIN32
    HANDLE in_write = nullptr;
    HANDLE out_read = nullptr;
    PROCESS_INFORMATION process{};
#else
    pid_t pid = -1;
    int write_fd = -1;
    int read_fd = -1;
#endif
    std::string pending;
};

static const char* pieceName(PieceType type) {
    switch (type) {
    case PieceType::S: return "S";
    case PieceType::Z: return "Z";
    case PieceType::J: return "J";
    case PieceType::L: return "L";
    case PieceType::T: return "T";
    case PieceType::O: return "O";
    case PieceType::I: return "I";
    default: throw std::runtime_error("no piece to name");
    }
}

static PieceType pieceType(const std::string& name) {
    for (PieceType type : { PieceType::S, PieceType::Z, PieceType::J, PieceType::L, PieceType::T, PieceType::O, PieceType::I }) {
        if (name == pieceName(type)) {
            return type;
        }
    }
    throw std::runtime_error("unknown piece " + name);
}

static RotationDirection rotation(const std::string& name) {
    if (name == "north") return RotationDirection::North;
    if (name == "east") return RotationDirection::East;
    if (name == "south") return RotationDirection::South;
    if (name == "west") return RotationDirection::West;
    throw std::runtime_error("unknown orientation " + name);
}

static spinType spin(const std::string& name) {
    if (name == "none") return spinType::null;
    if (name == "mini") return spinType::mini;
    if (name == "full") return spinType::normal;
    throw std::runtime_error("unknown spin " + name);
}

static const char* spinName(spinType s) {
    return s == spinType::normal ? "full" : s == spinType::mini ? "mini" : "none";
}

static nlohmann::json boardJson(const Board& board) {
    nlohmann::json rows = nlohmann::json::array();
    for (int y = 0; y < 40; y++) {
        nlohmann::json row = nlohmann::json::array();
        for (int x = 0; x < Board::width; x++) {
            if (y < (int)Board::height && board.get(x, y)) {
                row.push_back("G");
            }
            else {
                row.push_back(nullptr);
            }
        }
        rows.push_back(row);
    }
    return rows;
}

static nlohmann::json playerJson(const EmulationGame& game) {
    nlohmann::json player;
    player["board"] = boardJson(game.game.board);
    player["hold"] = game.game.hold ? nlohmann::json(pieceName(*game.game.hold)) : nlohmann::json(nullptr);
    player["queue"] = nlohmann::json::array({ pieceName(game.game.current_piece.type) });
    for (PieceType piece : game.game.queue) {
        player["queue"].push_back(pieceName(piece));
    }
    player["combo"] = (int)game.game.combo;
    player["back_to_back"] = (int)game.game.b2b;
    int meter = 0;
    for (int garbage : game.garbage_meter) {
        meter += garbage;
    }
    player["meter"] = meter;
    return player;
}

static nlohmann::json opponentJson() {
    nlohmann::json player;
    player["board"] = boardJson(Board{});
    player["hold"] = nullptr;
    player["queue"] = nlohmann::json::array();
    player["combo"] = 0;
    player["back_to_back"] = 0;
    player["meter"] = 0;
    return player;
}

static bool send_bag = true;

static nlohmann::json startJson(const EmulationGame& game) {
    nlohmann::json start = playerJson(game);
    start["type"] = "start";
    if (!send_bag) {
        start["opponents"] = nlohmann::json::array({ opponentJson() });
        return start;
    }
    nlohmann::json bag = nlohmann::json::array();
    const auto left_begin = game.rng.bag.begin() + game.rng.bagiterator;
    for (PieceType type : { PieceType::S, PieceType::Z, PieceType::J, PieceType::L, PieceType::T, PieceType::O, PieceType::I }) {
        if (left_begin >= game.rng.bag.end() || std::find(left_begin, game.rng.bag.end(), type) != game.rng.bag.end()) {
            bag.push_back(pieceName(type));
        }
    }
    start["randomizer"] = { { "type", "seven_bag" }, { "bag_state", bag } };
    start["opponents"] = nlohmann::json::array({ opponentJson() });
    return start;
}

static std::array<std::pair<int, int>, 4> cells(const Piece& piece) {
    std::array<std::pair<int, int>, 4> out;
    for (size_t i = 0; i < 4; i++) {
        out[i] = { piece.position.x + piece.minos[i].x, piece.position.y + piece.minos[i].y };
    }
    std::sort(out.begin(), out.end());
    return out;
}

static bool legalPlacement(const EmulationGame& game, const nlohmann::json& suggested, Piece& out) {
    const nlohmann::json& location = suggested["location"];
    const PieceType type = pieceType(location["type"].get<std::string>());
    const Piece wanted(type, rotation(location["orientation"].get<std::string>()),
                       Coord{ (int8_t)location["x"].get<int>(), (int8_t)location["y"].get<int>() });
    const spinType claimed = suggested.contains("spin") ? spin(suggested["spin"].get<std::string>()) : spinType::null;
    const auto target = cells(wanted);
    const spinType order[] = { spinType::normal, spinType::mini, spinType::null };
    bool found = false;
    int found_rank = 99;
    for (const Piece& placement : game.placements()) {
        if (placement.type != type || cells(placement) != target) {
            continue;
        }
        int rank = 0;
        while (order[rank] != placement.spin) {
            rank++;
        }
        const int claimed_rank = claimed == spinType::normal ? 0 : claimed == spinType::mini ? 1 : 2;
        if (rank >= claimed_rank && rank < found_rank) {
            out = placement;
            found_rank = rank;
            found = true;
        }
        else if (!found && rank < claimed_rank) {
            out = placement;
            out.spin = spinType::null;
            found_rank = 2;
            found = true;
        }
    }
    return found;
}

struct Options {
    std::string bot;
    uint64_t thinks = 0;
    int ms = 100;
    int games = 10;
    uint64_t seed = 0;
    int max_pieces = 2000;
    int threads = 1;
    int cpu = -1;
    int ramp = 0;
    float garbage_scale = 1;
};

static bool parse(int argc, const char** argv, Options& options) {
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string arg = argv[i];
        const std::string value = argv[i + 1];
        if (arg == "--bot") options.bot = value;
        else if (arg == "--no-bag") send_bag = std::stoi(value) == 0;
        else if (arg == "--thinks") options.thinks = std::stoull(value);
        else if (arg == "--ms") options.ms = std::stoi(value);
        else if (arg == "--games") options.games = std::stoi(value);
        else if (arg == "--seed") options.seed = std::stoull(value);
        else if (arg == "--max-pieces") options.max_pieces = std::stoi(value);
        else if (arg == "--threads") options.threads = std::stoi(value);
        else if (arg == "--cpu") options.cpu = std::stoi(value);
        else if (arg == "--ramp") options.ramp = std::stoi(value);
        else if (arg == "--garbage-scale") options.garbage_scale = std::stof(value);
        else return false;
    }
    return argc % 2 == 1 && !options.bot.empty() && options.games > 0 && options.ms > 0 && options.threads > 0;
}

struct GameResult {
    int pieces = 0;
    bool died = false;
    bool illegal = false;
    double think_seconds = 0;
};

static nlohmann::json receiveType(BotProcess& bot, const std::string& type) {
    while (true) {
        nlohmann::json message = nlohmann::json::parse(bot.receive());
        if (message.value("type", "") == type) {
            return message;
        }
        if (message.value("type", "") == "error") {
            throw std::runtime_error("the bot reports: " + message.dump());
        }
    }
}

static GameResult playGame(BotProcess& bot, EmulationGame& game, const Options& options) {
    using clock = std::chrono::steady_clock;
    GameResult result;
    bot.send(startJson(game).dump());

    while (true) {
        if (Shaktris::Utility::collides(game.game.board, game.game.current_piece) || game.placements().empty()) {
            result.died = true;
            break;
        }
        if (game.ramp_pieces >= options.max_pieces) {
            break;
        }

        const auto asked = clock::now();
        if (options.thinks == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(options.ms));
        }
        bot.send(R"({"type":"suggest"})");
        const nlohmann::json suggestion = receiveType(bot, "suggestion");
        result.think_seconds += std::chrono::duration<double>(clock::now() - asked).count();

        Piece piece(PieceType::I);
        if (!suggestion.contains("moves") || suggestion["moves"].empty()) {
            result.died = true;
            break;
        }
        const nlohmann::json& suggested = suggestion["moves"][0];
        if (!legalPlacement(game, suggested, piece)) {
            std::cerr << "illegal suggestion: " << suggested.dump() << std::endl;
            result.died = true;
            result.illegal = true;
            break;
        }

        Game expected = game.game;
        expected.place_piece(piece);
        expected.board.clearLines();

        game.set_move(Move(piece, false));
        game.play_moves();
        const int dealt = (int)std::count(game.game.queue.begin(), game.game.queue.end(), PieceType::Empty);
        game.chance_move();

        if (!(game.game.board == expected.board)) {
            bot.send(startJson(game).dump());
            continue;
        }
        nlohmann::json play;
        play["type"] = "play";
        play["move"] = suggested;
        play["move"]["spin"] = spinName(piece.spin);
        play["opponents"] = nlohmann::json::array({ opponentJson() });
        bot.send(play.dump());
        for (int i = (int)game.game.queue.size() - dealt; i < (int)game.game.queue.size(); i++) {
            bot.send(nlohmann::json({ { "type", "new_piece" }, { "piece", pieceName(game.game.queue[i]) } }).dump());
        }
    }

    result.pieces = game.ramp_pieces;
    return result;
}

int main(int argc, const char** argv) {
    Options options;
    try {
        if (!parse(argc, argv, options)) {
            throw std::invalid_argument("bad arguments");
        }
    }
    catch (const std::exception&) {
        std::cerr << "usage: " << argv[0]
            << " --bot PATH [--thinks N | --ms T] [--games K] [--seed S] [--max-pieces P] [--threads N]"
            << " [--cpu K] [--ramp R] [--garbage-scale X]" << std::endl;
        return 1;
    }
    if (options.cpu >= 0 && !restrictProcessToCpu(options.cpu)) {
        std::cerr << "can't restrict the process to cpu " << options.cpu << std::endl;
        return 1;
    }

    BotProcess bot;
    try {
        bot.start(options.bot);
        const nlohmann::json info = receiveType(bot, "info");
        const nlohmann::json features = info.value("features", nlohmann::json::array());
        auto has = [&](const char* feature) {
            return std::find(features.begin(), features.end(), feature) != features.end();
        };
        nlohmann::json rules;
        rules["type"] = "rules";
        rules["garbage_delay"] = EmulationGame::default_garbage_delay();
        if (has("set_threads")) {
            rules["threads"] = options.threads;
        }
        if (options.thinks > 0) {
            if (!has("set_thinks")) {
                throw std::runtime_error(info.value("name", "the bot") + " doesn't advertise set_thinks");
            }
            rules["thinks"] = options.thinks;
        }
        bot.send(rules.dump());
        receiveType(bot, "ready");

        int deaths = 0;
        long long total_pieces = 0;
        double total_think = 0;
        for (int i = 0; i < options.games; i++) {
            const uint64_t game_seed = options.seed + i;
            if (i > 0) {
                bot.send(R"({"type":"stop"})");
            }
            EmulationGame game = newGame(game_seed, options.ramp, options.garbage_scale);
            const GameResult result = playGame(bot, game, options);
            deaths += result.died;
            total_pieces += result.pieces;
            total_think += result.think_seconds;

            const int moves = std::max(result.pieces, 1);
            std::cout << "{\"seed\":" << game_seed
                << ",\"rules\":\"" << EmulationGame::rules() << "\""
                << ",\"bot\":" << nlohmann::json(info.value("name", "")).dump()
                << ",\"thinks\":" << options.thinks
                << ",\"ms\":" << (options.thinks ? 0 : options.ms)
                << ",\"ramp\":" << options.ramp
                << ",\"max_pieces\":" << options.max_pieces
                << ",\"pieces\":" << result.pieces
                << ",\"died\":" << (result.died ? "true" : "false")
                << ",\"illegal\":" << (result.illegal ? "true" : "false")
                << ",\"attack\":" << game.attack
                << ",\"true_attack\":" << game.true_attack
                << ",\"lines\":" << game.lines
                << ",\"opponent_attack\":" << game.markov.total_damage
                << ",\"ms_per_move\":" << 1000.0 * result.think_seconds / moves
                << "}" << std::endl;
        }
        bot.send(R"({"type":"quit"})");
        bot.finish();

        std::cerr << EmulationGame::rules() << " rules, " << info.value("name", "bot") << ", games " << options.games
            << ", died " << deaths << ", mean pieces " << (double)total_pieces / options.games
            << ", ms/move " << (total_pieces ? 1000.0 * total_think / total_pieces : 0.0) << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "nana_emulate_tbp: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
