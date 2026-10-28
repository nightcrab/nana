#include "MarkovOpponent.hpp"
#include "inference.hpp"
#include "Util/json.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main(int argc, char** argv) try {
    require(argc == 2, "usage: nana_neuralcheck TORCH_CASES.json");
    const opponent_inference::Model model(NANA_NEURAL_MODEL_DIR);
    std::ifstream input(argv[1]);
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
    auto& forecast = neural_opponent::forecast;
    forecast.active = true;
    forecast.cdf = {0.2f, 0.6f, 1.0f};
    for (auto& seq : forecast.sequences) seq = {4, 0, 6, 8, 8, 8, 8, 8};
    RootOpponent root;
    root.meter = 3;
    MarkovOpponent opponent;
    opponent.adopt(root);
    opponent.seed(42);
    opponent.resample_from_root();
    require(opponent.play() == 1 && opponent.last_attack == 4 && opponent.meter == 0,
            "neural attack must cancel the opponent's pending meter first");
    opponent.receiveAttack(2);
    require(opponent.play() == 0, "zero attack must preserve timing");
    opponent.receiveAttack(0);
    require(opponent.play() == 6, "neural amount must not use HMM bucket averages");
    require(opponent.play() == 0 && opponent.is_dead() && opponent.deaths == 1, "missing terminal death");
    for (int i = 0; i < 12; ++i) require(opponent.play() == 0 && opponent.deaths == 1, "neural death resurrected");
    for (int k = 0; k < 3; ++k) forecast.sequences[k].fill((uint8_t)(k+1));
    std::array<int, 3> counts{};
    opponent.seed(123);
    for (int n = 0; n < 20000; ++n) {
        opponent.adopt(RootOpponent{});
        opponent.resample_from_root();
        const int attack = (int)opponent.play();
        require(attack >= 1 && attack <= 3, "invalid sampled trajectory");
        ++counts[attack-1];
    }
    require(std::abs(counts[0]/20000.0-0.2) < 0.02 && std::abs(counts[1]/20000.0-0.4) < 0.02,
            "trajectory selection does not follow probability weights");
    forecast.active = false;
    std::cout << "Passed Torch beam parity on " << cases.size()
              << " positions; cancellation, timing, weighted selection and absorbing death\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
