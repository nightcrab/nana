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

namespace Search {

    extern bool searching;

    extern int core_count;

    extern int monte_carlo_depth;

    extern UCT uct;

    extern EmulationGame root_state;

    extern std::vector<std::unique_ptr<mpsc<Msg>>> queues;

    extern bool initialised;

    extern int time;

    struct FlushStats {
        uint64_t searches = 0;
        uint64_t changed = 0;
        double hidden_share = 0;
    };
    extern FlushStats flush_stats;

    void startSearch(const EmulationGame& state, int core_count);

    void continueSearch(EmulationGame state);

    void endSearch();

    extern uint64_t simulation_limit;

    extern const opponent::Root* root_opponent;
    extern const opponent::Belief* root_belief;
    bool waitForSimulations(std::chrono::milliseconds timeout);

    void maybeInsertNode(UCTNode node, const int threadIdx);

    // Opp is opponent::IOHMM, opponent::Neural or opponent::None, chosen once per search from config "opponent"
    template <class Opp>
    void search(std::stop_token stop, int threadIdx);

    template <class Opp>
    void processJob(const int threadIdx, Msg&& msg);

    template <class Opp>
    Reward rollout(EmulationGame& state, int threadIdx, size_t depth, float parent_board);

    void printStatistics();

    // See the best move found so far.
    Move bestMove();

};  // namespace Search
#endif