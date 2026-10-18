#include "Search.hpp"

#include <bit>
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

std::vector<std::unique_ptr<mpsc<Job>>> Search::queues;
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
// a rollout (expansion + eval) is about 30 messages' worth on the bench profile;
// the thresholds below are in the same units, so they can be tuned together
constexpr int MESSAGE_WEIGHT = 1;
#ifndef NANA_ROLLOUT_WEIGHT
#define NANA_ROLLOUT_WEIGHT 30
#endif
constexpr int ROLLOUT_WEIGHT = NANA_ROLLOUT_WEIGHT;
// ask for work once our queued work drops below this; we ask before running dry,
// so the answer arrives while we're still busy
#ifndef NANA_LOW_WATER
#define NANA_LOW_WATER 30
#endif
constexpr int LOW_WATER = NANA_LOW_WATER;
// give a rollout away if, without it, we'd still have this much more work than the requester.
// 0 = strict: we must keep a whole rollout's worth more, so we need two pending to give one
#ifndef NANA_GIVE_MARGIN
#define NANA_GIVE_MARGIN 3
#endif
constexpr int GIVE_MARGIN = NANA_GIVE_MARGIN == 0 ? ROLLOUT_WEIGHT : NANA_GIVE_MARGIN;
// after a REQUEST comes back unanswered, go round the loop this many times before asking again
constexpr int REQUEST_BACKOFF_ITERATIONS = 64;

// state private to one worker
struct alignas(64) WorkerLocal {
    // rollouts for new leaves we own, run when there are no messages or given away on request
    std::vector<Job> pending;
    // last load each thread told us about, piggybacked on their messages (work units)
    std::vector<int> load_view;
    bool request_outstanding = false;
    // we ran out of work since our last REQUEST was sent
    bool ran_dry = false;
    // loop iterations left before we may request again
    int request_backoff = 0;
};

static std::vector<WorkerLocal> worker_local;

static void resetWorkerLocal(int core_count) {
    worker_local = std::vector<WorkerLocal>(core_count);
    for (auto& local : worker_local) {
        local.load_view.assign(core_count, 0);
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
        queues.push_back(std::make_unique<mpsc<Job>>(core_count + 1));
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
        queues[rootOwnerIdx]->enqueue(Job(root_state, SELECT), core_count);
    }

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
        queues.emplace_back(std::make_unique<mpsc<Job>>(core_count + 1));
    }

    core_indices = std::vector<int>(core_count);

    std::iota(core_indices.begin(), core_indices.end(), 0);

    int rootOwnerIdx = uct.getOwner(state.hash());
    for (int j = 0; j < core_count; j++) {
        for (int i = 0; i < LOAD_FACTOR; i++) {
            root_state.opponent.reset_rng();
            root_state.rng.PPTRNG = fast_seed();
            queues[rootOwnerIdx]->enqueue(Job(root_state, SELECT), core_count);
        }

    }

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
        queues[i]->enqueue(Job(), core_count);
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
    std::cout << "duplicate expansions: " << duplicate_puts << std::endl;

    WorkerStatistics sum;
    for (const WorkerStatistics& s : uct.stats) {
        sum.requests += s.requests;
        sum.forwards += s.forwards;
        sum.gave += s.gave;
        sum.gave_hops += s.gave_hops;
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
    std::cout << "late answers: " << sum.late << std::endl;
#endif

#ifdef NANA_BENCH
    // per-thread time split; wait includes polling and flushing the queues
    // rollout (expansion + eval) runs inside select and expand jobs, so it overlaps those columns
    std::printf("thread    wait  select  backprop  expand   put | rollout |   jobs/s | ns/job: select backprop expand    put rollout\n");
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
        std::printf("%6d  %5.1f%%  %5.1f%%    %5.1f%%  %5.1f%% %5.1f%% |  %5.1f%% | %8.0f |      %6.0f   %6.0f %6.0f %6.0f %7.0f\n",
            i, pct(s.wait_cycles), pct(s.job_cycles[SELECT]), pct(s.job_cycles[BACKPROP]),
            pct(s.job_cycles[EXPAND]), pct(s.job_cycles[PUT]), pct(s.rollout_cycles), jobs / (ms / 1e6),
            ns(SELECT), ns(BACKPROP), ns(EXPAND), ns(PUT), rollout_ns);
    }
#endif
}

// how much work this thread already has queued, in work units
static int myLoad(int threadIdx) {
    return (int)worker_local[threadIdx].pending.size() * ROLLOUT_WEIGHT
        + (int)Search::queues[threadIdx]->buffered() * MESSAGE_WEIGHT;
}

// every message carries the sender's load, so receivers learn everyone's load for free
static void send(int from, int to, Job&& job) {
    job.sender = (uint16_t)from;
    job.sender_load = myLoad(from);
    Search::queues[to]->enqueue(std::move(job), from);
}

static uint64_t allThreadsMask() {
    return Search::core_count >= 64 ? ~0ull : (1ull << Search::core_count) - 1;
}

// a uniformly random thread from the mask; repeating this with the visited
// threads removed walks the threads in a random order (a random simple cycle)
static int randomThreadIn(uint64_t mask, int threadIdx) {
    int count = std::popcount(mask);
    int nth = Search::uct.rng[threadIdx].getRand(count);
    for (int i = 0; i < 64; i++) {
        if (mask & (1ull << i)) {
            if (nth-- == 0) return i;
        }
    }
    return -1;
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
    if (load >= LOW_WATER) {
        return;
    }

    // first hop: whoever looks richest; if nobody clearly is, a random thread
    int target = -1;
    int best = load + ROLLOUT_WEIGHT;
    for (int i = 0; i < Search::core_count; i++) {
        if (i != threadIdx && local.load_view[i] > best) {
            best = local.load_view[i];
            target = i;
        }
    }
    if (target < 0) {
        target = randomThreadIn(allThreadsMask() & ~(1ull << threadIdx), threadIdx);
    }

    Job request(REQUEST);
    request.requester = threadIdx;
    request.requester_load = load;
    request.visited = (1ull << threadIdx) | (1ull << target);

    local.request_outstanding = true;
    local.ran_dry = false;
    Search::uct.stats[threadIdx].requests++;
    send(threadIdx, target, std::move(request));
}

// give a rollout if that evens out the load, otherwise pass the request on
static void handleRequest(int threadIdx, Job&& request) {
    WorkerLocal& local = worker_local[threadIdx];
    int requester = request.requester;

    // without the rollout we give, we must still have GIVE_MARGIN more work than the requester,
    // so work only flows downhill and can't bounce back and forth
    int requester_load = request.requester_load;
    if (!local.pending.empty() && myLoad(threadIdx) - ROLLOUT_WEIGHT >= requester_load + GIVE_MARGIN) {
        // give away the oldest one
        Job rollout = std::move(local.pending.front());
        local.pending.erase(local.pending.begin());
        rollout.is_reply = true;

        auto& stats = Search::uct.stats[threadIdx];
        stats.gave++;
        stats.gave_hops += std::popcount(request.visited) - 1;
        send(threadIdx, requester, std::move(rollout));
        return;
    }

    uint64_t unvisited = allThreadsMask() & ~request.visited;
    if (unvisited == 0) {
        // everyone was asked
        request.type = NO_WORK;
        send(threadIdx, requester, std::move(request));
        return;
    }

    int next = randomThreadIn(unvisited, threadIdx);
    request.visited |= 1ull << next;
    Search::uct.stats[threadIdx].forwards++;
    send(threadIdx, next, std::move(request));
}

void Search::processJob(const int threadIdx, Job&& job) {

    if (job.type == PUT) {
        if (!uct.insertNode(std::move(*job.node))) {
            uct.stats[threadIdx].duplicate_puts++;
        }
        return;
    }

    if (job.type == REQUEST) {
        handleRequest(threadIdx, std::move(job));
        return;
    }

    if (job.type == NO_WORK) {
        // nobody had a rollout to spare
        WorkerLocal& local = worker_local[threadIdx];
        local.request_outstanding = false;
        local.request_backoff = REQUEST_BACKOFF_ITERATIONS;
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

        send(threadIdx, parentIdx, std::move(job));
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

            send(threadIdx, parentIdx, std::move(job));

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
            Action* action = &node.actions[0];

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

            if constexpr (search_style == NANA) {
                action = &node.select(depth);
            }
            if constexpr (search_style == CC) {

                action = &node.select_SOR(uct.rng[threadIdx]);
            }

            // Virtual Loss by setting N := N+1
            node.N += 1;

            action->addN();

            action->updateTime(time);

            // remember how this choice looked, and where we were, in case we restart here
            if constexpr (NANA_MPMCTS) {
                job.history.push_back(node.historyRow(hash, *action, depth));
                job.snapshots.push_back(state);
            }

            state.set_move(state.specific_move(action->move));

            state.play_moves();
            state.chance_move();

            uint32_t new_hash = state.hash();

            // Get the owner of the updated state based on the hash
            uint32_t ownerIdx = uct.getOwner(new_hash);

            // reuse this job for the next select step
            job.path.push_back(HashActionPair(hash, action->id));

            uct.stats[threadIdx].deepest_node = std::max(uct.stats[threadIdx].deepest_node, (uint64_t)job.path.size());

            send(threadIdx, ownerIdx, std::move(job));
        }
        else {
            // a new leaf we own: queue its rollout so an idle thread can ask for it.
            // whoever runs it sends the reward to the parent's owner
            uct.stats[threadIdx].deferred++;
            job.type = EXPAND;
            worker_local[threadIdx].pending.push_back(std::move(job));
        }
    }
    else if (job.type == BACKPROP) {

        uct.stats[threadIdx].backprop_messages++;
        UCTNode& node = uct.getNode(job.path.back().hash, threadIdx);
        Action& edge = node.actions[job.path.back().actionID];

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

            if (threadIdx == uct.getOwner(root_state.hash())) {
                // extra scope to prevent notifying the main thread while holding the lock
                // this is to prevent the main thread from waiting on the condition variable while we're holding the lock
                {
                    std::scoped_lock lock(min_work_mutex);
                    min_work_bool = true;
                }
                min_work_cv.notify_one();
                // notify the main thread that we're done
            }

            // give ourself this job
            queues[threadIdx]->enqueue(std::move(job), threadIdx);

            return;
        }

        if constexpr (!NANA_MPMCTS) {
            send(threadIdx, uct.getOwner(job.path.back().hash), std::move(job));
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
            send(threadIdx, threadIdx, std::move(job));
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

        send(threadIdx, uct.getOwner(job.path.back().hash), std::move(job));
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
        send(threadIdx, owner, Job(std::move(node), PUT));
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
            for (int i = 0; i < JOB_TYPE_COUNT; i++) {
                out.job_cycles[i] = local.job_cycles[i];
                out.job_count[i] = local.job_count[i];
            }
        }
    } profile{ uct.stats[threadIdx] };
#endif

    WorkerLocal& local = worker_local[threadIdx];

    auto run = [&](Job&& job, [[maybe_unused]] uint64_t start) {
#ifdef NANA_BENCH
        const JobType type = job.type;
#endif
        processJob(threadIdx, std::move(job));
#ifdef NANA_BENCH
        profile.local.job_cycles[type] += __rdtsc() - start;
        profile.local.job_count[type]++;
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
            Job job = queues[threadIdx]->dequeue();
            if (job.sender < core_count) {
                local.load_view[job.sender] = job.sender_load;
            }
            if (job.type == STOP) {
                return;
            }
            run(std::move(job), t0);
        }
        else if (!local.pending.empty()) {
            // our own newest rollout, while its state is still in cache
            Job job = std::move(local.pending.back());
            local.pending.pop_back();
            run(std::move(job), t0);
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

    for (auto& action : node.actions) {
        max_eval = std::max(max_eval, action.eval);
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

    for (Action& action : uct.getNode(root_state.hash()).actions) {
        if constexpr (search_style == NANA) {
            if (action.N == biggest_N) {
                if (action.R > biggest_R) {
                    biggest_R = action.R;
                    biggest_N = action.N;
                    best_move = action.move;
                }
            }
            if (action.N > biggest_N) {
                biggest_R = action.R;
                biggest_N = action.N;
                best_move = action.move;
            }
#ifndef TBP
           //std::cout << "N:" << action.N << " R_avg:" << action.Q() << std::endl;
#endif
        }
        if constexpr (search_style == CC) {
            if (action.R > biggest_R) {
                biggest_R = action.R;
                biggest_N = action.N;
                best_move = action.move;
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