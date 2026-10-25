#pragma once
// NANA_CC_LOCKSTEP (nana_cctree only): the CC search takes its selection draws from a splitmix64
// stream that Cold Clear's tree_dump can replay (its `lockstep` feature), and logs every selection,
// so the two trees can be compared step by step
#include <cstdint>
#include <vector>

#ifndef NANA_CC_LOCKSTEP
#define NANA_CC_LOCKSTEP 0
#endif

class UCTNode;

namespace cc_trace {

struct Candidate {
    uint32_t edge;
    float R;
    double weight;
};

// one select_CC call: the node, its live edges best first, the draw and the rank it picked
struct Step {
    uint32_t node;
    std::vector<Candidate> ranked;
    double u;
    uint32_t pick;
};

inline uint64_t state = 0;
inline std::vector<Step> steps;

// replaces select_CC when set: nana_cctree's --canonical ranking, which Cold Clear's tree_dump
// mirrors (ties broken by occupied cells, weights from integer score units)
inline size_t (*select_hook)(const UCTNode&) = nullptr;

inline void seed(uint64_t s) {
    state = s;
    steps.clear();
}

// uniform in [0, 1), 53 bits
inline double uniform() {
    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}

}  // namespace cc_trace
