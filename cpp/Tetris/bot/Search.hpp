#pragma once
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <memory>

#include "EmulationGame.hpp"
#include "Util/MPSC.hpp"
#include "Move.hpp"
#include "UCT.hpp"

#ifndef __SEARCH_HPP
#define __SEARCH_HPP

enum SearchType {
    CC,
    NANA
};


namespace Search {

    extern bool searching;

    constexpr SearchType search_style = NANA;

    extern int core_count;

    extern int monte_carlo_depth;

    extern UCT uct;

    extern EmulationGame root_state;

    extern std::vector<std::unique_ptr<mpsc<Msg>>> queues;

    extern bool initialised;

    extern int time;

    // NANA_FLUSH_AT_END: at the end of each search, what the parked MP-MCTS buffers were hiding
    // from the root's statistics, which bestMove reads
    struct FlushStats {
        uint64_t searches = 0;
        uint64_t changed = 0;       // searches where the hidden visits would change the best move
        double hidden_share = 0;    // share of the root's visits that were hidden, summed over searches
    };
    extern FlushStats flush_stats;

    void startSearch(const EmulationGame& state, int core_count);

    void continueSearch(EmulationGame state);

    void endSearch();

    void maybeInsertNode(UCTNode node, const int threadIdx);

    void search(std::stop_token stop, int threadIdx);

    void processJob(const int threadIdx, Msg&& msg);

    float rollout(EmulationGame& state, int threadIdx);

    void printStatistics();

    // See the best move found so far.
    Move bestMove();

};  // namespace Search
#endif