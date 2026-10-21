#pragma once

#include <atomic>
#include <cstdint>
#include <random>

// cheap seeds for reseeding simulation RNGs.
// RNG::new_seed() constructs a std::random_device per call, which on Linux can mean a
// syscall or a slow, chip-wide RDSEED; the search reseeds on every simulation restart,
// which made the root's owner a serial bottleneck. this draws from a per-thread
// splitmix64 that only touches std::random_device once per thread.
//
// fast_seed_reset(seed) makes it deterministic: every thread restarts from seed, mixed with the
// order threads first draw after the reset. with one search thread (and the main thread), a game
// then replays exactly, which tuner/ relies on to compare configs on the same games.
// fast_seed_reset(0) goes back to random_device.
namespace fast_seed_detail {
inline std::atomic<uint64_t> base{ 0 };
inline std::atomic<uint64_t> epoch{ 0 };
inline std::atomic<uint64_t> threads{ 0 };

inline uint64_t mix(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
}

inline void fast_seed_reset(uint64_t seed) {
    using namespace fast_seed_detail;
    base.store(seed);
    threads.store(0);
    epoch.fetch_add(1);
}

inline uint32_t fast_seed() {
    using namespace fast_seed_detail;
    thread_local uint64_t state = 0;
    thread_local uint64_t seen_epoch = ~0ull;

    const uint64_t now = epoch.load(std::memory_order_relaxed);
    if (seen_epoch != now) {
        seen_epoch = now;
        const uint64_t b = base.load(std::memory_order_relaxed);
        if (b != 0) {
            state = mix(b + 0x9E3779B97F4A7C15ull * (1 + threads.fetch_add(1)));
        }
        else {
            state = (uint64_t(std::random_device{}()) << 32) ^ std::random_device{}();
        }
    }

    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    return uint32_t(mix(z) >> 32);
}
