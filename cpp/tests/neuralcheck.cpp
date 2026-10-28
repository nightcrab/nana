#include "EmulationGame.hpp"
#include "inference.hpp"
#include "Util/json.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main(int argc, char** argv) try {
    require(argc == 3, "usage: nana_neuralcheck MODEL_DIR TORCH_CASES.json");
    const opponent_inference::Model model(argv[1]);
    std::ifstream input(argv[2]);
    const auto cases = nlohmann::json::parse(input);
    for (const auto& test : cases) {
        const auto board = test.at("board").get<std::vector<float>>();
        const auto context = test.at("context").get<std::vector<float>>();
        require(board.size() == 200 && context.size() == 52, "invalid parity input dimensions");
        const auto actual = opponent_inference::trajectories(model, board.data(), context.data());
        for (int k = 0; k < 3; ++k) {
            const auto expected = test.at("predictions")[k];
            require(nlohmann::json(actual[k].sequence) == expected.at("sequence"), "beam trajectory differs from Torch");
            require(std::abs(actual[k].logp-expected.at("logp").get<float>()) < 0.001f,
                    "absolute beam log probability differs from Torch");
        }
    }

    using opponent::Neural;
    auto& forecast = opponent::forecast;
    forecast.active = true;
    forecast.cdf = {0.2f, 0.6f, 1.0f};
    for (auto& seq : forecast.sequences) seq = {4, 0, 6, 8, 8, 8, 8, 8};
    EmulationGame game(false);
    opponent::Root root;
    root.meter = 3;
    game.adopt_opponent(root);
    game.opponent.rng.seed(42);
    game.neural.sample_root(game.opponent.rng);
    require(game.opponent_attack<Neural>() == 1 && game.opponent.last_attack == 4 && game.opponent.meter == 0,
            "neural attack must cancel the opponent's pending meter first");
    game.opponent_receive(2);
    require(game.opponent_attack<Neural>() == 0, "zero attack must preserve timing");
    game.opponent_receive(0);
    require(game.opponent_attack<Neural>() == 6, "neural amount must not use HMM bucket averages");
    require(game.opponent_attack<Neural>() == 0 && game.opponent_dead() && game.opponent.deaths == 1,
            "missing terminal death");
    for (int i = 0; i < 12; ++i) {
        require(game.opponent_attack<Neural>() == 0 && game.opponent.deaths == 1, "neural death resurrected");
    }

    for (auto& seq : forecast.sequences) seq.fill(1);
    game.adopt_opponent(opponent::Root{});
    game.neural.sample_root(game.opponent.rng);
    for (int i = 0; i < opponent::Forecast::HORIZON; ++i) {
        require(game.opponent_attack<Neural>() == 1, "missing forecast attack within the horizon");
    }
    for (int i = 0; i < 12; ++i) {
        require(game.opponent_attack<Neural>() == 0 && !game.opponent_dead(), "attack predicted past the horizon");
    }

    for (int k = 0; k < 3; ++k) forecast.sequences[k].fill((uint8_t)(k+1));
    std::array<int, 3> counts{};
    game.opponent.rng.seed(123);
    for (int n = 0; n < 20000; ++n) {
        game.adopt_opponent(opponent::Root{});
        game.neural.sample_root(game.opponent.rng);
        const int attack = (int)game.opponent_attack<Neural>();
        require(attack >= 1 && attack <= 3, "invalid sampled trajectory");
        ++counts[attack-1];
    }
    require(std::abs(counts[0]/20000.0-0.2) < 0.02 && std::abs(counts[1]/20000.0-0.4) < 0.02,
            "trajectory selection does not follow probability weights");
    forecast.active = false;
    std::cout << "Passed Torch beam parity on " << cases.size()
              << " positions; cancellation, timing, weighted selection, absorbing death and the horizon\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
