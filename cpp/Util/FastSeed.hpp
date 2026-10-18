#pragma once

#include <cstdint>
#include <random>

// cheap seeds for reseeding simulation RNGs.
// RNG::new_seed() constructs a std::random_device per call, which on Linux can mean a
// syscall or a slow, chip-wide RDSEED; the search reseeds on every simulation restart,
// which made the root's owner a serial bottleneck. this draws from a per-thread
// splitmix64 that only touches std::random_device once per thread.
inline uint32_t fast_seed() {
    thread_local uint64_t state = (uint64_t(std::random_device{}()) << 32) ^ std::random_device{}();

    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return uint32_t((z ^ (z >> 31)) >> 32);
}
