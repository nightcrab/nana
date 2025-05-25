#pragma once


#include <algorithm>
#include <chrono>
#include <memory>
#include <unordered_map>
#include <vector>

#include "EmulationGame.hpp"
#include "Move.hpp"
#include "UCT.hpp"
#include "Util/MPSC.hpp"
#include <stop_token>

// replacement of the search namespace

enum SearchType {
      CC,NANA
};
constexpr SearchType search_style = NANA;

class Nana {

    UCT uct;
    EmulationGame root_state;
    std::vector<std::unique_ptr<mpsc<JobVariant>>> queues;
    std::chrono::high_resolution_clock::time_point start_search_time;
    std::vector<std::jthread> worker_threads;
    std::stop_source worker_stopper;
    // for continue search
    int core_count = 0;
    int time = 0;

public:
    bool searching = false;

    void startSearch(const EmulationGame& state, int core_count);
    void continueSearch(const EmulationGame& state);
    void endSearch();
    void printStatistics();
    Move bestMove();
};