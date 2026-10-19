#include "Search.hpp"

#include <bit>
#include <cmath>
#include <execution>
#include <numeric>
#include <ranges>

#include "Util/Affinity.hpp"
#include "Util/Distribution.hpp"
#include "Eval.hpp"
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#ifdef NANA_BENCH
// time spent in rollout (expansion + eval), which runs inside SELECT and EXPAND jobs
static thread_local uint64_t tl_rollout_cycles = 0;
static thread_local uint64_t tl_rollout_count = 0;
#endif

// used to be an atomic for stopping the threads
bool Search::searching = false;

// stop source for stopping the threads
// this is used to stop the threads when the main thread wants to stop searching
std::stop_source thread_stopper;

int Search::core_count = 0;

int Search::monte_carlo_depth = 1;

UCT Search::uct;

EmulationGame Search::root_state;

// stuff for notifying the main thread that the worker threads have at least a minimum amount of work done
std::mutex min_work_mutex;
std::condition_variable min_work_cv;
std::atomic_bool min_work_bool;

std::vector<std::unique_ptr<mpsc<Msg>>> Search::queues;
std::vector<int> core_indices;
std::vector<std::jthread> worker_threads;

std::chrono::steady_clock::time_point search_start_time;

bool Search::initialised = false;
int Search::time = 0;

// jobs in flight per worker
#ifndef NANA_LOAD_FACTOR
#define NANA_LOAD_FACTOR 6
#endif
constexpr int LOAD_FACTOR = NANA_LOAD_FACTOR;

// selects past this depth are treated as leaves; a safety net for cycles in the state graph
constexpr size_t MAX_PATH_DEPTH = 64;

// 1 = MP-MCTS: backprops stop and restart selection where every ancestor would choose the same path.
// 0 = TDS-UCT: every backprop goes to the root
#ifndef NANA_MPMCTS
#define NANA_MPMCTS 1
#endif

// load balancing is lock-free work requesting: each node is only touched by its owner,
// so the only work that can move between threads is a rollout (expansion + eval).
// owners queue rollouts for new leaves, and threads running low ask others for one.

// work requesting: queued work is scored in work units, a queued message being 1.
// the thresholds are in the same units, so they can be tuned together.
// they only decide which thread runs a rollout and when, not what the search computes,
// so they're safe to tune for throughput. each can be overridden by an environment
// variable (NANA_LOW_WATER=60 etc., see loadTuning), read when a search starts.
struct RequestTuning {
    // a queued message
    int message_weight = 1;
    // a rollout (expansion + eval): about 30 messages' worth on the bench profile
    int rollout_weight = 30;
    // ask for work once our queued work drops below this; we ask before running dry,
    // so the answer arrives while we're still busy
    int low_water = 30;
    // give a rollout away if, without it, we'd still have this much more work than the requester
    int give_margin = 3;
    // after a REQUEST comes back unanswered, go round the loop this many times before asking again
    int backoff_iterations = 64;

    // who to ask: score each thread by the load it last reported, minus this penalty if it
    // shares a different L3 cache (chiplet) than the requester, since the rollout's data
    // would then cross chiplets; keep the best steal_topk and sample one with Gumbel-max
    // at steal_temperature (0 = always the best)
    int steal_chiplet_penalty = 30;
    int steal_topk = 4;
    int steal_temperature = 10;
};

constexpr int MAX_STEAL_TOPK = 8;

static RequestTuning tuning;

static void loadTuning() {
    auto read = [](const char* name, int& value) {
        if (const char* text = std::getenv(name)) {
            value = std::atoi(text);
        }
    };
    tuning = RequestTuning();
    read("NANA_MESSAGE_WEIGHT", tuning.message_weight);
    read("NANA_ROLLOUT_WEIGHT", tuning.rollout_weight);
    read("NANA_LOW_WATER", tuning.low_water);
    read("NANA_GIVE_MARGIN", tuning.give_margin);
    read("NANA_BACKOFF_ITERATIONS", tuning.backoff_iterations);
    read("NANA_STEAL_CHIPLET_PENALTY", tuning.steal_chiplet_penalty);
    read("NANA_STEAL_TOPK", tuning.steal_topk);
    read("NANA_STEAL_TEMPERATURE", tuning.steal_temperature);
    tuning.steal_topk = std::clamp(tuning.steal_topk, 1, MAX_STEAL_TOPK);
}

// state private to one worker
struct alignas(64) WorkerLocal {
    // rollouts for new leaves we own, run when there are no messages or given away on request
    std::vector<Msg> pending;
    // last load each thread told us about, piggybacked on their messages (work units)
    std::vector<int> load_view;
    bool request_outstanding = false;
    // we ran out of work since our last REQUEST was sent
    bool ran_dry = false;
    // loop iterations left before we may request again
    int request_backoff = 0;
};

static std::vector<WorkerLocal> worker_local;

// which L3 cache (chiplet) each worker's core shares; workers are pinned by index
static std::vector<int> worker_l3_group;

static void resetWorkerLocal(int core_count) {
    worker_local = std::vector<WorkerLocal>(core_count);
    for (auto& local : worker_local) {
        local.load_view.assign(core_count, 0);
    }
    worker_l3_group.resize(core_count);
    for (int i = 0; i < core_count; i++) {
        worker_l3_group[i] = cacheGroupOfCore(i);
    }
}

void Search::startSearch(const EmulationGame &state, int core_count) {
    //std::cout << "started searching, root hash is: " <<  (int) (state.hash() % 1000) << std::endl;

    search_start_time = std::chrono::steady_clock::now();

    searching = true;
    // reset stop bool for min wait time
    min_work_bool = false;
    // reset the thread stopper
    thread_stopper = std::stop_source();

    Search::core_count = core_count;

    root_state = state;

    time = 0;

    // Create new search tree
    uct = UCT(core_count);

    // Create root node
    uct.insertNode(UCTNode(root_state));

    // Initialise worker queues

    queues.clear();
    queues.reserve(core_count);

    for (int i = 0; i < core_count; i++) {
        queues.push_back(std::make_unique<mpsc<Msg>>(core_count + 1));
    }

    // Thread indices
    core_indices = std::vector<int>(core_count);
    worker_threads = std::vector<std::jthread>(core_count);

    std::iota(core_indices.begin(), core_indices.end(), 0);

    int rootOwnerIdx = uct.getOwner(state.hash());

    // Spawn jobs
    for (int i = 0; i < LOAD_FACTOR * core_count; i++) {
        root_state.rng.PPTRNG = fast_seed();
        root_state.opponent.reset_rng();
        queues[rootOwnerIdx]->enqueue(std::make_unique<Job>(root_state, SELECT), core_count);
    }

    loadTuning();
    resetWorkerLocal(core_count);

    // Spawn worker threads
    for (auto& idx : core_indices) {
        worker_threads[idx] = std::jthread(search, thread_stopper.get_token(), idx);
    }

    initialised = true;
};

void Search::continueSearch(EmulationGame state) {
    //std::cout << "started searching, root hash is: " << (int)(state.hash() % 1000) << std::endl;

    if (state.game_over) {
        return;
    }

    search_start_time = std::chrono::steady_clock::now();

    searching = true;
    // reset stop bool for min wait time
    min_work_bool = false;
    // reset the thread stopper
    thread_stopper = std::stop_source();

    root_state = state;

    time = time + 1;

    root_state.attack = state.app() * 0;
    root_state.true_attack = state.true_app() * 0;
    root_state.pieces = 0;
    root_state.lines = 2;
    root_state.opponent.deaths = 0;
    root_state.opponent = Opponent();

    if (!uct.nodeExists(state.hash())) {
        uct.insertNode(UCTNode(state));
    }

    for (WorkerStatistics& stat : uct.stats) {
        stat = {};
    }
    queues.clear();
    queues.reserve(core_count);
    // Initialise worker queues
    for (int i = 0; i < core_count; i++) {
        queues.emplace_back(std::make_unique<mpsc<Msg>>(core_count + 1));
    }

    core_indices = std::vector<int>(core_count);

    std::iota(core_indices.begin(), core_indices.end(), 0);

    int rootOwnerIdx = uct.getOwner(state.hash());
    for (int j = 0; j < core_count; j++) {
        for (int i = 0; i < LOAD_FACTOR; i++) {
            root_state.opponent.reset_rng();
            root_state.rng.PPTRNG = fast_seed();
            queues[rootOwnerIdx]->enqueue(std::make_unique<Job>(root_state, SELECT), core_count);
        }

    }

    loadTuning();
    resetWorkerLocal(core_count);

    for (auto& idx : core_indices) {
        worker_threads[idx] = std::jthread(search, thread_stopper.get_token(), idx);
    }
}

void Search::endSearch() {
    if (!min_work_bool) {
        std::unique_lock<std::mutex> lock(min_work_mutex);
        min_work_cv.wait(lock, [] { return min_work_bool.load(); });
    }

    searching = false;
    thread_stopper.request_stop();

    for (int i = 0; i < core_count; i++) {
        queues[i]->enqueue(std::make_unique<Job>(), core_count);
    }

    // join threads
    for (auto& thread : worker_threads) {
		thread.join();
	}

    // reset stop bool for min wait time
    min_work_bool = false;
    // reset the thread stopper
    thread_stopper = std::stop_source();

    // garbage collect
    if (uct.map_size() > 200000) {
        uct.collect();
    }

    queues.clear();
    // rollouts still pending are dropped with the rest of the in-flight jobs
    worker_local.clear();

    //std::cout << "stopped searching" << std::endl;
 };

void Search::printStatistics() {

    std::chrono::steady_clock::time_point search_end_time = std::chrono::steady_clock::now();

    double ms = std::chrono::duration_cast<std::chrono::microseconds>(search_end_time - search_start_time).count();

    uint64_t nodes = 0;
    uint64_t depth = 0;
    uint64_t backprops = 0;
    uint64_t duplicate_puts = 0;

    for (WorkerStatistics stat : uct.stats) {
        nodes += stat.nodes;
        backprops += stat.backprop_messages;
        depth = std::max(stat.deepest_node, depth);
        duplicate_puts += stat.duplicate_puts;
    }

	std::cout << "nodes: " << nodes << std::endl;
    std::cout << "nodes / second: " << nodes / (ms/ 1000000) << std::endl;
    std::cout << "backprops / second: " << backprops / (ms / 1000000) << std::endl;
    std::cout << "tree depth: " << depth << std::endl;

#ifdef NANA_BENCH
    std::cout << "tuning: rollout_weight " << tuning.rollout_weight << " low_water " << tuning.low_water
        << " give_margin " << tuning.give_margin << " backoff_iterations " << tuning.backoff_iterations
        << " steal_chiplet_penalty " << tuning.steal_chiplet_penalty << " steal_topk " << tuning.steal_topk
        << " steal_temperature " << tuning.steal_temperature << std::endl;
    {
        int groups = 0;
        for (int g : worker_l3_group) groups = std::max(groups, g + 1);
        std::cout << "L3 cache groups across workers: " << groups << std::endl;
    }
    std::cout << "duplicate expansions: " << duplicate_puts << std::endl;

    WorkerStatistics sum;
    for (const WorkerStatistics& s : uct.stats) {
        sum.requests += s.requests;
        sum.forwards += s.forwards;
        sum.gave += s.gave;
        sum.gave_hops += s.gave_hops;
        sum.gave_cross_l3 += s.gave_cross_l3;
        sum.nones += s.nones;
        sum.received += s.received;
        sum.late += s.late;
        sum.deferred += s.deferred;
        sum.restarts += s.restarts;
        sum.restart_depth += s.restart_depth;
        sum.root_backprops += s.root_backprops;
        sum.cycles += s.cycles;
        sum.depth_capped += s.depth_capped;
    }

    // MP-MCTS
    uint64_t history_nodes = 0;
    uint64_t history_rows = 0;
    for (auto* side : { &uct.nodes_left, &uct.nodes_right }) {
        for (auto& shard : *side) {
            for (auto& [id, n] : shard.obj_) {
                if (!n.history.empty()) {
                    history_nodes++;
                    history_rows += n.history.size();
                }
            }
        }
    }
    std::cout << "restarts: " << sum.restarts << std::endl;
    std::cout << "backprops stopped early: " << (backprops ? 100.0 * sum.restarts / backprops : 0.0) << "%" << std::endl;
    std::cout << "mean restart depth: " << (sum.restarts ? (double)sum.restart_depth / sum.restarts : 0.0) << std::endl;
    std::cout << "root backprops: " << sum.root_backprops << std::endl;
    std::cout << "cycles: " << sum.cycles << std::endl;
    std::cout << "depth capped: " << sum.depth_capped << std::endl;
    std::cout << "nodes storing history: " << history_nodes << " (" << history_rows * sizeof(HistoryRow) / 1024 << " KiB)" << std::endl;
    std::cout << "deferred rollouts: " << sum.deferred << std::endl;
    // every simulation ends in exactly one rollout, so this counts simulations even when
    // MP-MCTS restarts skip select steps (which makes nodes / second incomparable)
    std::cout << "simulations / second: " << sum.deferred / (ms / 1000000) << std::endl;
    std::cout << "requests: " << sum.requests << std::endl;
    std::cout << "answered: " << sum.gave << std::endl;
    std::cout << "unanswered: " << sum.nones << std::endl;
    std::cout << "forwards: " << sum.forwards << std::endl;
    std::cout << "mean threads asked when answered: " << (sum.gave ? (double)sum.gave_hops / sum.gave : 0.0) << std::endl;
    std::cout << "stolen rollouts: " << (sum.deferred ? 100.0 * sum.gave / sum.deferred : 0.0) << "% of all, "
        << (sum.gave ? 100.0 * sum.gave_cross_l3 / sum.gave : 0.0) << "% of those across L3 caches" << std::endl;
    std::cout << "late answers: " << sum.late << std::endl;
#endif

#ifdef NANA_BENCH
    // per-thread time split; wait includes polling and flushing the queues
    // rollout (expansion + eval) runs inside select and expand jobs, so it overlaps those columns
    // queue us: how long messages waited in this thread's queue; top: share of this thread's
    // time spent selecting at the root or depth 1
    std::printf("thread    wait  select  backprop  expand   put | rollout |   jobs/s | ns/job: select backprop expand    put rollout | queue us   top\n");
    for (int i = 0; i < (int)uct.stats.size(); i++) {
        const WorkerStatistics& s = uct.stats[i];
        uint64_t total = s.wait_cycles;
        for (int t = 0; t < JOB_TYPE_COUNT; t++) total += s.job_cycles[t];
        if (total == 0) continue;
        uint64_t jobs = 0;
        for (int t = 0; t < JOB_TYPE_COUNT; t++) jobs += s.job_count[t];
        auto pct = [&](uint64_t c) { return 100.0 * c / total; };
        // the thread was alive for the whole search, so total cycles ~ ms
        auto ns = [&](int t) { return s.job_count[t] ? (double)s.job_cycles[t] / total * ms * 1000.0 / s.job_count[t] : 0.0; };
        double rollout_ns = s.rollout_count ? (double)s.rollout_cycles / total * ms * 1000.0 / s.rollout_count : 0.0;
        double queue_us = s.queue_count ? (double)s.queue_cycles / total * ms / s.queue_count : 0.0;
        std::printf("%6d  %5.1f%%  %5.1f%%    %5.1f%%  %5.1f%% %5.1f%% |  %5.1f%% | %8.0f |      %6.0f   %6.0f %6.0f %6.0f %7.0f | %8.1f %4.1f%%\n",
            i, pct(s.wait_cycles), pct(s.job_cycles[SELECT]), pct(s.job_cycles[BACKPROP]),
            pct(s.job_cycles[EXPAND]), pct(s.job_cycles[PUT]), pct(s.rollout_cycles), jobs / (ms / 1e6),
            ns(SELECT), ns(BACKPROP), ns(EXPAND), ns(PUT), rollout_ns, queue_us, pct(s.top_cycles));
    }
#endif
}

// how much work this thread already has queued, in work units
static int myLoad(int threadIdx) {
    return (int)worker_local[threadIdx].pending.size() * tuning.rollout_weight
        + (int)Search::queues[threadIdx]->buffered() * tuning.message_weight;
}

// every message carries the sender's load, so receivers learn everyone's load for free
static void send(int from, int to, Msg&& msg) {
    msg->sender = (uint16_t)from;
    msg->sender_load = myLoad(from);
#ifdef NANA_BENCH
    msg->sent_at = __rdtsc();
#endif
    Search::queues[to]->enqueue(std::move(msg), from);
}

static uint64_t allThreadsMask() {
    return Search::core_count >= 64 ? ~0ull : (1ull << Search::core_count) - 1;
}

// pick which thread to ask for a rollout on the requester's behalf, from the candidates mask,
// using the chooser's picture of everyone's load. score = reported load, minus a penalty for
// a different L3 cache than the requester; keep the top k; then Gumbel-max sample among them,
// i.e. pick with probability proportional to exp(score / temperature).
// asking only unvisited threads each hop still visits every thread at most once
static int chooseVictim(int chooser, int requester, uint64_t candidates) {
    const WorkerLocal& local = worker_local[chooser];
    const int n = Search::core_count;

    struct Candidate {
        float score;
        int thread;
    };
    Candidate top[MAX_STEAL_TOPK];
    int count = 0;

    // start from a random thread so equal scores (e.g. all 0 at startup) aren't always
    // resolved towards the lowest indices
    int start = (int)(fast_seed() % (uint32_t)n);
    for (int step = 0; step < n; step++) {
        int j = (start + step) % n;
        if (!(candidates & (1ull << j))) continue;

        float score = (float)local.load_view[j];
        if (worker_l3_group[j] != worker_l3_group[requester]) {
            score -= (float)tuning.steal_chiplet_penalty;
        }

        // insert into the small top-k list, kept sorted best first
        int i;
        if (count < tuning.steal_topk) {
            i = count++;
        }
        else if (score > top[count - 1].score) {
            // replaces the worst one kept
            i = count - 1;
        }
        else {
            continue;
        }
        while (i > 0 && top[i - 1].score < score) {
            top[i] = top[i - 1];
            i--;
        }
        top[i] = { score, j };
    }
    if (count == 0) {
        return -1;
    }
    if (tuning.steal_temperature <= 0) {
        return top[0].thread;
    }

    int chosen = top[0].thread;
    float best = -INFINITY;
    for (int i = 0; i < count; i++) {
        // standard Gumbel noise from a uniform in (0, 1)
        float u = ((float)(fast_seed() >> 8) + 0.5f) * (1.0f / 16777216.0f);
        float value = top[i].score / (float)tuning.steal_temperature - std::log(-std::log(u));
        if (value > best) {
            best = value;
            chosen = top[i].thread;
        }
    }
    return chosen;
}

// ask for a rollout before we run out, so the answer lands while we're still busy
static void maybeRequest(int threadIdx) {
    WorkerLocal& local = worker_local[threadIdx];
    if (local.request_outstanding || Search::core_count < 2) {
        return;
    }
    if (local.request_backoff > 0) {
        local.request_backoff--;
        return;
    }
    int load = myLoad(threadIdx);
    if (load >= tuning.low_water) {
        return;
    }

    int target = chooseVictim(threadIdx, threadIdx, allThreadsMask() & ~(1ull << threadIdx));

    Msg request = std::make_unique<Job>(REQUEST);
    request->requester = threadIdx;
    request->requester_load = load;
    request->visited = (1ull << threadIdx) | (1ull << target);

    local.request_outstanding = true;
    local.ran_dry = false;
    Search::uct.stats[threadIdx].requests++;
    send(threadIdx, target, std::move(request));
}

// give a rollout if that evens out the load, otherwise pass the request on
static void handleRequest(int threadIdx, Msg&& msg) {
    Job& request = *msg;
    WorkerLocal& local = worker_local[threadIdx];
    int requester = request.requester;

    // without the rollout we give, we must still have give_margin more work than the requester,
    // so work only flows downhill and can't bounce back and forth
    int requester_load = request.requester_load;
    if (!local.pending.empty() && myLoad(threadIdx) - tuning.rollout_weight >= requester_load + tuning.give_margin) {
        // give away the oldest one
        Msg rollout = std::move(local.pending.front());
        local.pending.erase(local.pending.begin());
        rollout->is_reply = true;

        auto& stats = Search::uct.stats[threadIdx];
        stats.gave++;
        stats.gave_hops += std::popcount(request.visited) - 1;
        if (worker_l3_group[threadIdx] != worker_l3_group[requester]) {
            stats.gave_cross_l3++;
        }
        send(threadIdx, requester, std::move(rollout));
        return;
    }

    uint64_t unvisited = allThreadsMask() & ~request.visited;
    if (unvisited == 0) {
        // everyone was asked
        request.type = NO_WORK;
        send(threadIdx, requester, std::move(msg));
        return;
    }

    // same choice as the first hop, from our own picture of the load, scored for the requester
    int next = chooseVictim(threadIdx, requester, unvisited);
    request.visited |= 1ull << next;
    Search::uct.stats[threadIdx].forwards++;
    send(threadIdx, next, std::move(msg));
}

void Search::processJob(const int threadIdx, Msg&& msg) {
    // the job lives on the heap and belongs to whoever holds msg: don't touch it after sending
    Job& job = *msg;

    if (job.type == PUT) {
        if (!uct.insertNode(std::move(*job.node))) {
            uct.stats[threadIdx].duplicate_puts++;
        }
        return;
    }

    if (job.type == REQUEST) {
        handleRequest(threadIdx, std::move(msg));
        return;
    }

    if (job.type == NO_WORK) {
        // nobody had a rollout to spare
        WorkerLocal& local = worker_local[threadIdx];
        local.request_outstanding = false;
        local.request_backoff = tuning.backoff_iterations;
        uct.stats[threadIdx].nones++;
        return;
    }

    if (job.type == EXPAND) {
        if (job.is_reply) {
            WorkerLocal& local = worker_local[threadIdx];
            local.request_outstanding = false;
            uct.stats[threadIdx].received++;
            if (local.ran_dry) {
                uct.stats[threadIdx].late++;
            }
            local.ran_dry = false;
            job.is_reply = false;
        }

        // a rollout task: it may not be our state, so don't look it up; treat it as a leaf.
        // rollout sends the new node to its owner, and the reward goes to the parent's owner
        uct.stats[threadIdx].nodes++;

        float reward = rollout(job.state, threadIdx);

        uint32_t parentIdx = uct.getOwner(job.path.back().hash);

        job.R = reward;
        job.extra_N = 0;
        job.depth = job.state.pieces;
        job.type = BACKPROP;

        send(threadIdx, parentIdx, std::move(msg));
        return;
    }

    if (job.type == SELECT) {

        uct.stats[threadIdx].nodes++;

        EmulationGame& state = job.state;

        int depth = state.pieces;

        uint32_t hash = state.hash();

        if (state.game_over) {

            float reward = rollout(state, threadIdx);

            if (job.path.empty()) {
                return;
            }

            uint32_t parent_hash = job.path.back().hash;

            uint32_t parentIdx = uct.getOwner(parent_hash);

            // reuse this job as the backprop job and send it back to our parent
            job.R = reward;
            job.extra_N = 0;
            job.depth = state.pieces;
            job.type = BACKPROP;

            send(threadIdx, parentIdx, std::move(msg));

            return;
        }

        bool exists = uct.nodeExists(hash);
        if (exists && job.path.size() >= MAX_PATH_DEPTH) {
            // safety net for cycles: treat it as a leaf
            uct.stats[threadIdx].depth_capped++;
            exists = false;
        }

        if (exists) {

            UCTNode& node = uct.getNode(hash, threadIdx);
            size_t action = 0;
            [[maybe_unused]] HistoryRow row;

#ifdef NANA_BENCH
            for (const HashActionPair& step : job.path) {
                if (step.hash == hash) {
                    uct.stats[threadIdx].cycles++;
                    break;
                }
            }
#endif

            // MP-MCTS: busy nodes keep the freshest history for the path that reached them
            if (NANA_MPMCTS && node.N >= HISTORY_MIN_VISITS) {
                node.history = job.history;
            }

            // MP-MCTS with PUCT: the select scan also collects the history row's competitors
            constexpr bool single_pass = NANA_MPMCTS && search_style == NANA;
            if constexpr (single_pass) {
                action = node.selectWithCompetitors(depth, row);
            }
            else if constexpr (search_style == NANA) {
                action = node.select(depth);
            }
            if constexpr (search_style == CC) {

                action = node.select_SOR(uct.rng[threadIdx]);
            }

            // Virtual Loss by setting N := N+1
            node.N += 1;

            EdgeStats& edge = node.stats(action);
            edge.addN();

            node.updateTime(action, time);

            // remember how this choice looked, and where we were, in case we restart here
            if constexpr (NANA_MPMCTS) {
                if constexpr (single_pass) {
                    // the competitors came from the select scan; the rest is after the virtual loss
                    row.hash = hash;
                    row.chosen = (uint32_t)action;
                    row.node_N = node.N;
                    row.depth = depth;
                    row.edges[0] = { edge.R, edge.N, edge.prior };
                    job.history.push_back(row);
                }
                else {
                    job.history.push_back(node.historyRow(hash, action, depth));
                }
                job.snapshots.push_back(state);
            }

            state.set_move(state.specific_move(node.move(action)));

            state.play_moves();
            state.chance_move();

            uint32_t new_hash = state.hash();

            // Get the owner of the updated state based on the hash
            uint32_t ownerIdx = uct.getOwner(new_hash);

            // reuse this job for the next select step
            job.path.push_back(HashActionPair(hash, (int)action));

            uct.stats[threadIdx].deepest_node = std::max(uct.stats[threadIdx].deepest_node, (uint64_t)job.path.size());

            send(threadIdx, ownerIdx, std::move(msg));
        }
        else {
            // a new leaf we own: queue its rollout so an idle thread can ask for it.
            // whoever runs it sends the reward to the parent's owner
            uct.stats[threadIdx].deferred++;
            job.type = EXPAND;
            worker_local[threadIdx].pending.push_back(std::move(msg));
        }
    }
    else if (job.type == BACKPROP) {

        uct.stats[threadIdx].backprop_messages++;
        UCTNode& node = uct.getNode(job.path.back().hash, threadIdx);
        EdgeStats& edge = node.stats(job.path.back().actionID);

        float reward = job.R;

        // Undo Virtual Loss by adding R
        if constexpr (search_style == NANA) {

            edge.addReward(reward);

        }
        if constexpr (search_style == CC) {
            if (reward > edge.R) {
                edge.R = reward;
            }
        }

        // simulations that restarted below us weren't counted on the way down
        edge.N += job.extra_N;
        node.N += job.extra_N;

        job.path.pop_back();

        if (job.path.empty()) {
            uct.stats[threadIdx].root_backprops++;

            root_state.opponent.reset_rng();
            root_state.rng.PPTRNG = fast_seed();

            // reuse this job for a fresh select from the root;
            // copy-assigning keeps the existing vector capacity
            job.state = root_state;
            job.R = 0.0;
            job.extra_N = 0;
            job.history.clear();
            job.snapshots.clear();
            job.type = SELECT;

            // tell the main thread a simulation has completed; it only needs telling once per
            // search, so later root backprops skip the mutex (and the root hash) entirely
            if (!min_work_bool.load(std::memory_order_relaxed)) {
                // extra scope to prevent notifying the main thread while holding the lock
                // this is to prevent the main thread from waiting on the condition variable while we're holding the lock
                {
                    std::scoped_lock lock(min_work_mutex);
                    min_work_bool = true;
                }
                min_work_cv.notify_one();
            }

            // give ourself this job
            queues[threadIdx]->enqueue(std::move(msg), threadIdx);

            return;
        }

        if constexpr (!NANA_MPMCTS) {
            send(threadIdx, uct.getOwner(job.path.back().hash), std::move(msg));
            return;
        }

        job.history.pop_back();
        EmulationGame here = std::move(job.snapshots.back());
        job.snapshots.pop_back();

        // MP-MCTS: would every ancestor still send a simulation down to us?
        // take the history rows for our path, preferring the fresher copies this node keeps,
        // and update them as if this reward had gone all the way up
        static thread_local std::vector<HistoryRow> rows;
        rows = job.history;
        for (size_t i = 0; i < rows.size() && i < node.history.size(); i++) {
            const HistoryRow& kept = node.history[i];
            if (kept.hash == rows[i].hash && kept.chosen == rows[i].chosen && kept.node_N > rows[i].node_N) {
                rows[i] = kept;
            }
        }

        bool still_chosen = true;
        for (HistoryRow& row : rows) {
            row.edges[0].R += reward;
            row.edges[0].N += job.extra_N;
            row.node_N += job.extra_N;
            still_chosen = still_chosen && row.stillChosen();
        }

        if (still_chosen) {
            // the ancestors would just send a simulation straight back here, so start it here.
            // park the reward for the next backprop that leaves this node, plus a visit for
            // the restarted simulation, which the ancestors haven't counted
            node.R_buffer += reward;
            node.N_buffer += job.extra_N + 1;

            // the restarted simulation is also a visit through every ancestor in the history,
            // otherwise their Q never moves and we'd restart here forever
            for (HistoryRow& row : rows) {
                row.edges[0].N += 1;
                row.node_N += 1;
            }

            uct.stats[threadIdx].restarts++;
            uct.stats[threadIdx].restart_depth += job.path.size();

            if (node.N >= HISTORY_MIN_VISITS) {
                node.history = rows;
            }

            here.rng.PPTRNG = fast_seed();
            here.opponent.reset_rng();

            job.state = std::move(here);
            job.history = rows;
            job.R = 0.0;
            job.extra_N = 0;
            job.type = SELECT;

            // we own this node, so the select starts with us
            send(threadIdx, threadIdx, std::move(msg));
            return;
        }

        if (node.N >= HISTORY_MIN_VISITS) {
            node.history = rows;
        }

        // carry up anything parked here
        job.R = reward + node.R_buffer;
        job.extra_N += node.N_buffer;
        node.R_buffer = 0;
        node.N_buffer = 0;

        send(threadIdx, uct.getOwner(job.path.back().hash), std::move(msg));
    }
}

void Search::maybeInsertNode(UCTNode node, const int threadIdx) {
    int owner = uct.getOwner(node.id);
    if (threadIdx == owner) {
        if (!uct.insertNode(std::move(node))) {
            uct.stats[threadIdx].duplicate_puts++;
        }
    }
    else {
        send(threadIdx, owner, std::make_unique<Job>(std::move(node), PUT));
    }
}

void Search::search(std::stop_token stop, const int threadIdx) {
    // one worker per physical core before any core gets two
    pinThreadToCore(threadIdx);

#ifdef NANA_BENCH
    // accumulate locally and publish on exit, so profiling doesn't add shared writes
    struct Profile {
        WorkerStatistics& out;
        WorkerStatistics local;
        ~Profile() {
            out.rollout_cycles = tl_rollout_cycles;
            out.rollout_count = tl_rollout_count;
            tl_rollout_cycles = tl_rollout_count = 0;
            out.wait_cycles = local.wait_cycles;
            out.queue_cycles = local.queue_cycles;
            out.queue_count = local.queue_count;
            out.top_cycles = local.top_cycles;
            out.top_count = local.top_count;
            for (int i = 0; i < JOB_TYPE_COUNT; i++) {
                out.job_cycles[i] = local.job_cycles[i];
                out.job_count[i] = local.job_count[i];
            }
        }
    } profile{ uct.stats[threadIdx] };
#endif

    WorkerLocal& local = worker_local[threadIdx];

    auto run = [&](Msg&& msg, [[maybe_unused]] uint64_t start) {
#ifdef NANA_BENCH
        const JobType type = msg->type;
        const bool top = type == SELECT && msg->path.size() <= 1;
#endif
        processJob(threadIdx, std::move(msg));
#ifdef NANA_BENCH
        uint64_t cycles = __rdtsc() - start;
        profile.local.job_cycles[type] += cycles;
        profile.local.job_count[type]++;
        if (top) {
            profile.local.top_cycles += cycles;
            profile.local.top_count++;
        }
#endif
    };

    while (true) {
        if (stop.stop_requested()) {
            return;
        }

        uint64_t t0 = 0;
#ifdef NANA_BENCH
        t0 = __rdtsc();
#endif
        bool has_message = !queues[threadIdx]->isempty();

        maybeRequest(threadIdx);

        // messages first: they're cheap, and other threads' simulations wait on them
        if (has_message) {
            Msg msg = queues[threadIdx]->dequeue();
            Job& job = *msg;
            if (job.sender < core_count) {
                local.load_view[job.sender] = job.sender_load;
            }
#ifdef NANA_BENCH
            if (job.sent_at) {
                // TSCs are synchronised across cores on current x86; clamp in case they aren't quite
                int64_t queued = (int64_t)(__rdtsc() - job.sent_at);
                profile.local.queue_cycles += queued > 0 ? queued : 0;
                profile.local.queue_count++;
                // jobs are reused; only time them again if they go through send()
                job.sent_at = 0;
            }
#endif
            if (job.type == STOP) {
                return;
            }
            run(std::move(msg), t0);
        }
        else if (!local.pending.empty()) {
            // our own newest rollout, while its state is still in cache
            Msg msg = std::move(local.pending.back());
            local.pending.pop_back();
            run(std::move(msg), t0);
        }
        else {
            local.ran_dry = true;
            _mm_pause();
#ifdef NANA_BENCH
            profile.local.wait_cycles += __rdtsc() - t0;
#endif
        }
    }
}


float Search::rollout(EmulationGame& state, int threadIdx) {
    // Rollout using eval.
#ifdef NANA_BENCH
    struct RolloutTimer {
        uint64_t start = __rdtsc();
        ~RolloutTimer() {
            tl_rollout_cycles += __rdtsc() - start;
            tl_rollout_count++;
        }
    } rollout_timer;
#endif

    float reward = 0;

    uct.stats[threadIdx].nodes++;

    if (state.game_over) {
        return -0.0;
    }

    UCTNode node(state);

    float max_eval = 0.0;

    for (size_t i = 0; i < node.actionCount(); i++) {
        max_eval = std::max(max_eval, node.eval(i));
    }

    maybeInsertNode(std::move(node), threadIdx);

    if constexpr (search_style == NANA) {
        float r = state.true_app() / 3 + max_eval / 2;
        reward = std::max(reward, r);
    }
    if constexpr (search_style == CC) {
        float r = state.true_app() / 3 + max_eval / 2;
        reward = std::max(reward, r);
    }

    if (state.opponent.garbage_height() > 15) {
        reward += state.opponent.garbage_height() / 20;
    }

    reward += state.opponent.deaths / 3;

    if (state.opponent.is_dead()) {
        // gottem
        reward = 1;
    }
    return reward;
}

Move Search::bestMove() {

    int biggest_N = 0;
    float biggest_R = 0;
    Move best_move;

    UCTNode& root = uct.getNode(root_state.hash());
    for (size_t i = 0; i < root.actionCount(); i++) {
        const EdgeStats& action = root.stats(i);
        if constexpr (search_style == NANA) {
            if (action.N == biggest_N) {
                if (action.R > biggest_R) {
                    biggest_R = action.R;
                    biggest_N = action.N;
                    best_move = root.move(i);
                }
            }
            if (action.N > biggest_N) {
                biggest_R = action.R;
                biggest_N = action.N;
                best_move = root.move(i);
            }
#ifndef TBP
           //std::cout << "N:" << action.N << " R_avg:" << action.Q() << std::endl;
#endif
        }
        if constexpr (search_style == CC) {
            if (action.R > biggest_R) {
                biggest_R = action.R;
                biggest_N = action.N;
                best_move = root.move(i);
            }

#ifndef TBP
            //std::cout << "N:" << action.N << " R_max:" << action.R << std::endl;
#endif
        }
        

    }
    if constexpr (search_style == NANA) {
#ifndef TBP
         //std::cout << "best move was visited " << biggest_N << " times, with R_avg " << biggest_R / biggest_N << std::endl;
#endif
    }
    if constexpr (search_style == CC) {
#ifndef TBP
         //std::cout << "best move was visited " << biggest_N << " times, with R_max " << biggest_R << std::endl;
#endif
    }

    return best_move;
}