#include "Neural.hpp"
#include "Config.hpp"
#include "inference.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace opponent {

using namespace opponent_inference;
static_assert(CONTEXT_SIZE == 52 && HORIZON == Forecast::HORIZON && BINS == Forecast::DEATH + 1);

namespace {
const Model& model() {
    static const Model loaded = [] {
        const std::string& path = config().neural_model;
        if (path.empty()) {
            throw std::runtime_error("the neural opponent needs config neural_model, the exported model directory");
        }
        return Model(path);
    }();
    return loaded;
}
}

void Neural::encode(const Board& seen, int combo, int b2b, int meter, PieceType hold,
                    const std::vector<PieceType>& queue) {
    forecast.active = false;
    if (queue.size() < 6) {
        throw std::runtime_error("the neural opponent needs the opponent's TBP2 board, hold and six visible pieces");
    }
    const Model& m = model();
    const auto started = std::chrono::steady_clock::now();
    std::array<float, CELLS> board{};
    std::array<float, CONTEXT_SIZE> context{};
    for (int y = 0; y < 20; ++y) for (int x = 0; x < 10; ++x) board[y*10+x] = seen.get(x, y);
    context[0] = std::log1p(std::max(0, combo));
    context[1] = std::log1p(std::max(0, b2b));
    context[2] = std::log1p(std::max(0, meter));
    // Same I,O,T,L,J,S,Z one-hot order as opponent_data.context().
    constexpr std::array<PieceType, 7> order{PieceType::I, PieceType::O, PieceType::T, PieceType::L,
                                           PieceType::J, PieceType::S, PieceType::Z};
    for (int token = 0; token < 7; ++token) {
        const PieceType piece = token == 0 ? hold : queue[token-1];
        for (int j = 0; j < 7; ++j) context[3+token*7+j] = piece == order[j] ? 1.0f : 0.0f;
    }
    const auto candidates = trajectories(m, board.data(), context.data());
    float total = 0;
    for (int k = 0; k < Forecast::TRAJECTORIES; ++k) {
        std::copy(candidates[k].sequence.begin(), candidates[k].sequence.end(), forecast.sequences[k].begin());
        total += std::exp(candidates[k].logp);
        forecast.cdf[k] = total;
    }
    forecast.retained_mass = total;
    for (float& p : forecast.cdf) p /= total;
    forecast.cdf.back() = 1;
    forecast.active = true;
    static unsigned count = 0;
    static double total_ms = 0;
    total_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-started).count();
    if (++count == 1 || count % 1000 == 0)
        std::cerr << "neural forecasts=" << count << " mean_ms=" << total_ms/count
                  << " top3_mass=" << total << std::endl;
}

}
