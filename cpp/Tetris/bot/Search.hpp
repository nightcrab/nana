#pragma once
#include <chrono>
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

    // stop the search's workers once this many simulations have run (0 = no limit). a worker checks
    // between jobs, so with one thread a search runs exactly this many, and with fast_seed_reset a
    // game replays exactly (tuner/ relies on this)
    extern uint64_t simulation_limit;

    // what's known of the real opponent (RootOpponent, from its board through the opponent encoder),
    // set by the caller before a search: the search's HMM opponent then carries on from it (a hidden
    // state and since counter drawn per simulation) instead of starting fresh at every root.
    // nullptr = start fresh
    extern const RootOpponent* root_opponent;
    // Caller's filter, copied into each simulation root.
    extern const OpponentFilter* root_filter;
    // wait until the search reaches simulation_limit, or for timeout; false on timeout
    bool waitForSimulations(std::chrono::milliseconds timeout);

    void maybeInsertNode(UCTNode node, const int threadIdx);

    void search(std::stop_token stop, int threadIdx);

    void processJob(const int threadIdx, Msg&& msg);

    Reward rollout(EmulationGame& state, int threadIdx, size_t depth, float parent_board);

#if !NANA_SUFFIX
    float leafReward(EmulationGame& state, float max_eval);
#endif

    void printStatistics();

    // See the best move found so far.
    Move bestMove();

};  // namespace Search
#endif