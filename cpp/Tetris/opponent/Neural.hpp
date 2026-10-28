#pragma once
#include "Opponent.hpp"
#include "engine/Board.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace opponent {

// The neural model's prediction at the root: its most likely attack sequences over the next HORIZON pieces.
// Published between searches; workers only read it until the search ends.
struct Forecast {
    static constexpr int TRAJECTORIES = 3;
    static constexpr int HORIZON = 8;
    // The bin for a death; the other bins are attack amounts.
    static constexpr int DEATH = 8;

    std::array<std::array<uint8_t, HORIZON>, TRAJECTORIES> sequences{};
    std::array<float, TRAJECTORIES> cdf{};
    float retained_mass = 0;
    bool active = false;
};
inline Forecast forecast;

// Replays one of the forecast's trajectories, then predicts no attack past its horizon. It needs no filter: the
// root is encoded afresh from the opponent's visible state every move.
class Neural {
public:
    static constexpr bool simulated = true;

    // Writes `forecast` from the opponent's board, hold and next six pieces. The model is loaded from config
    // "neural_model" on first use.
    static void encode(const Board& board, int combo, int b2b, int meter, PieceType hold,
                       const std::vector<PieceType>& queue);

    Prediction predict(float, float, Rng&) {
        if (step >= Forecast::HORIZON) {
            return {};
        }
        const int bin = forecast.sequences[trajectory][step++];
        if (bin == Forecast::DEATH) {
            return { 0, true };
        }
        return { (float)bin, false };
    }

    // A death in the forecast is final.
    bool respawn() { return false; }

    void sample_root(Rng& rng) {
        trajectory = (uint8_t)rng.sample(forecast.cdf);
        step = forecast.active ? 0 : Forecast::HORIZON;
    }

private:
    uint8_t trajectory = 0;
    uint8_t step = 0;
};

}
