#pragma once
#include <array>
#include <cstdint>

// Published between searches; workers only read it until the search ends.
namespace neural_opponent {
struct Forecast {
    std::array<std::array<uint8_t, 8>, 3> sequences{};
    std::array<float, 3> cdf{};
    float retained_mass = 0;
    bool active = false;
};
inline Forecast forecast;
}
