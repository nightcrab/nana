#include "Search.hpp"
#include "Config.hpp"
#include <algorithm>
#include <array>
#include <cstring>

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

EmulationGame Search::root_state(false);

// stuff for notifying the main thread that the worker threads have at least a minimum amount of work done
std::mutex min_work_mutex;
std::condition_variable min_work_cv;
std::atomic_bool min_work_bool;

uint64_t Search::simulation_limit = 0;
const opponent::Root* Search::root_opponent = nullptr;
const opponent::Belief* Search::root_belief = nullptr;
static std::mutex budget_mutex;
static std::condition_variable budget_cv;
static std::atomic_bool budget_reached;

static uint64_t simulationsSoFar() {
    uint64_t total = 0;
    for (const WorkerStatistics& s : Search::uct.stats) {
        total += s.rollouts;
    }
    return total;
}

bool Search::waitForSimulations(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(budget_mutex);
    return budget_cv.wait_for(lock, timeout, [] { return budget_reached.load(); });
}

std::vector<std::unique_ptr<mpsc<Msg>>> Search::queues;
std::vector<int> core_indices;
std::vector<std::jthread> worker_threads;

std::chrono::steady_clock::time_point search_start_time;

bool Search::initialised = false;
int Search::time = 0;
Search::FlushStats Search::flush_stats;

#ifndef NANA_LOAD_FACTOR
#define NANA_LOAD_FACTOR 6
#endif
constexpr int LOAD_FACTOR = NANA_LOAD_FACTOR;

constexpr size_t MAX_PATH_DEPTH = 64;
static size_t leafDepth() {
    static const size_t depth = [] {
        const int cap = cc_search ? 0 : config().depth_cap;
        return cap > 0 && (size_t)cap < MAX_PATH_DEPTH ? (size_t)cap : MAX_PATH_DEPTH;
    }();
    return depth;
}

#ifndef NANA_MPMCTS
#define NANA_MPMCTS !NANA_CC_SEARCH
#endif
static_assert(!cc_search || !NANA_MPMCTS,
              "the CC search needs NANA_MPMCTS=0: the MP-MCTS path selects with PUCT and restarts on PUCT's "
              "stillChosen test; the CC search gets its restart from NANA_CC_RESTART instead");


static EmulationGame replay_root(false);
static uint32_t replay_root_hash = 0;

template <class Opp>
[[maybe_unused]] static size_t replayPath(const Job& job, uint32_t restart_hash, EmulationGame& state) {
    state = replay_root;
    state.start_from_root<Opp>();
    const size_t depth = job.path.size();
    for (size_t i = 0; i < depth; i++) {
        state.set_move(state.specific_move(job.moves[i].unpack()));
        state.play_moves<Opp>();
        state.chance_move<Opp>();
        const uint32_t expected = i + 1 < depth ? job.path[i + 1].hash : restart_hash;
        if (state.game_over || state.hash() != expected) {
            return i + 1;
        }
    }
    return depth + 1;
}

struct RequestTuning {
    int message_weight = 1;
    int rollout_weight = 30;
    int low_water = 30;
    int give_margin = 3;
    int backoff_iterations = 64;

    int steal_chiplet_penalty = 30;
    int steal_topk = 4;
    int steal_temperature = 10;
};

constexpr int MAX_STEAL_TOPK = 8;

static RequestTuning tuning;

static void loadTuning() {
    const NanaConfig& c = config();
    tuning.message_weight = c.message_weight;
    tuning.rollout_weight = c.rollout_weight;
    tuning.low_water = c.low_water;
    tuning.give_margin = c.give_margin;
    tuning.backoff_iterations = c.backoff_iterations;
    tuning.steal_chiplet_penalty = c.steal_chiplet_penalty;
    tuning.steal_topk = std::clamp(c.steal_topk, 1, MAX_STEAL_TOPK);
    tuning.steal_temperature = c.steal_temperature;
}

#ifndef NANA_OPPONENT_ORACLE
#define NANA_OPPONENT_ORACLE 0
#endif

// config "opponent", read once per search: picks which compiled search the workers run
enum class SearchOpponent { None, IOHMM, Neural };
static SearchOpponent search_opponent = SearchOpponent::None;

// Calls f.template operator()<Opp>() with the opponent type the search models.
template <class F>
static decltype(auto) withOpponent(F&& f) {
    switch (search_opponent) {
    case SearchOpponent::IOHMM:
        return f.template operator()<opponent::IOHMM>();
    case SearchOpponent::Neural:
        return f.template operator()<opponent::Neural>();
    default:
        return f.template operator()<opponent::None>();
    }
}

static void loadSearchOpponent() {
    const std::string& name = config().opponent;
    search_opponent = name == "iohmm" ? SearchOpponent::IOHMM
                    : name == "neural" ? SearchOpponent::Neural
                    : SearchOpponent::None;
    if (search_opponent == SearchOpponent::Neural && !opponent::forecast.active) {
        std::cerr << "the neural opponent needs a forecast of the real opponent, which only nana_tbp encodes"
                  << std::endl;
        std::exit(2);
    }
    if constexpr (NANA_OPPONENT_ORACLE || NANA_FUTURE_ORACLE) {
        if (search_opponent != SearchOpponent::IOHMM) {
            std::cerr << "the opponent and future oracles need opponent iohmm" << std::endl;
            std::exit(2);
        }
    }
}

static void startFromRoot(EmulationGame& state) {
    withOpponent([&]<class Opp>() { state.start_from_root<Opp>(); });
}

static void adoptOpponentBelief(EmulationGame& root) {
    opponent::root_prior.active = false;
    if (Search::root_opponent) {
        root.adopt_opponent(*Search::root_opponent);
        Search::root_opponent->publish();
    }
}

#ifndef NANA_PHANTOM_GARBAGE
#define NANA_PHANTOM_GARBAGE 0
#endif
static void searchGarbage(EmulationGame& root) {
    if constexpr (NANA_FUTURE_ORACLE) {
        static_assert(!NANA_FUTURE_ORACLE || !cc_search, "future oracle requires the regular search");
        EmulationGame::deterministic_chance = true;
    }
    if constexpr (cc_search) {
        EmulationGame::deterministic_chance = false;
        withOpponent([&]<class Opp>() { root.reseed_chance<Opp>(); });
        EmulationGame::deterministic_chance = true;
    }
    if constexpr (NANA_PHANTOM_GARBAGE || cc_search) {
        root.garbage_scale = 0;
        root.phantom_garbage = NANA_PHANTOM_GARBAGE && !cc_search && search_opponent != SearchOpponent::None;
        root.phantom_meter.clear();
    }
}

static void searchOpponent(EmulationGame& root) {
    if (search_opponent == SearchOpponent::None) {
        opponent::root_prior.active = false;
        root.track_belief = false;
        return;
    }
    if constexpr (NANA_OPPONENT_ORACLE || NANA_FUTURE_ORACLE) {
        opponent::root_prior.active = false;
    }
    else {
        root.reset_opponent();
        adoptOpponentBelief(root);
    }
    root.track_belief = search_opponent == SearchOpponent::IOHMM && Search::root_belief != nullptr;
    if (root.track_belief) {
        root.belief = *Search::root_belief;
    }
}

#ifndef NANA_BELIEF_LOG
#define NANA_BELIEF_LOG 0
#endif
#if NANA_BELIEF_LOG
static void logBelief(const EmulationGame& state, size_t depth, float surprise) {
    static std::mutex mutex;
    static std::FILE* file = [] {
        const char* path = std::getenv("NANA_BELIEF_LOG_FILE");
        std::FILE* f = path && *path ? std::fopen(path, "w") : nullptr;
        if (f) std::fputs("depth,surprise,sent,combo,b2b,meter,garbage,q0,q1,q2,q3,q4,q5,q6,q7\n", f);
        return f;
    }();
    if (!file || !state.track_belief) {
        return;
    }
    const auto q = state.belief.predicted();
    std::lock_guard<std::mutex> lock(mutex);
    int meter = 0;
    for (int garbage : state.garbage_meter) {
        meter += garbage;
    }
    std::fprintf(file, "%zu,%.6g,%d,%d,%d,%d,%d", depth, surprise * 10000.0f, state.last_sent, (int)state.game.combo,
                 (int)state.game.b2b, meter, (int)state.game.board.get_garbage_height());
    for (float p : q) {
        std::fprintf(file, ",%.4g", p);
    }
    std::fputc('\n', file);
}
#endif

struct alignas(64) WorkerLocal {
    std::vector<Msg> pending;
    std::vector<int> load_view;
    bool request_outstanding = false;
    bool ran_dry = false;
    int request_backoff = 0;
};

static std::vector<WorkerLocal> worker_local;

// Per owner, ids of nodes that may hold parked visits, so the end-of-search
// flush touches only those nodes, never the whole tree. Entries can repeat or
// go stale; the flush dedups and filters them.
static std::vector<std::vector<uint32_t>> parked_ids;

static void noteParked(int threadIdx, const UCTNode& node) {
    if (node.N_buffer == 0) {
        parked_ids[threadIdx].push_back(node.id);
    }
}

// Lookup that doesn't promote an old-generation node, unlike UCT::getNode.
static UCTNode* findNode(uint32_t id) {
    for (auto* side : { &Search::uct.nodes_right, &Search::uct.nodes_left }) {
        auto& map = (*side)[id % Search::uct.workers].obj_;
        auto it = map.find(id);
        if (it != map.end()) {
            return &it->second;
        }
    }
    return nullptr;
}

struct alignas(64) DepthBaseline {
    std::array<double, MAX_PATH_DEPTH + 1> sum{};
    std::array<double, MAX_PATH_DEPTH + 1> count{};

    float mean(size_t d) const { return count[d] > 0 ? (float)(sum[d] / count[d]) : 0.0f; }

    float correct(size_t d, float raw, float parent_board) {
        d = std::min(d, MAX_PATH_DEPTH);
        sum[d] += raw - parent_board;
        count[d] += 1;
        float drift = 0;
        for (size_t k = 1; k <= d; k++) {
            drift += mean(k);
        }
        return raw - drift;
    }

    void shift() {
        for (size_t d = 0; d < MAX_PATH_DEPTH; d++) {
            sum[d] = sum[d + 1] * 0.5;
            count[d] = count[d + 1] * 0.5;
        }
        sum[MAX_PATH_DEPTH] = count[MAX_PATH_DEPTH] = 0;
    }
};
static std::vector<DepthBaseline> depth_baseline;

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

using Worker = void (*)(std::stop_token, int);

static Worker worker() {
    return withOpponent([]<class Opp>() -> Worker { return &Search::search<Opp>; });
}

void Search::startSearch(const EmulationGame &state, int core_count) {
    //std::cout << "started searching, root hash is: " <<  (int) (state.hash() % 1000) << std::endl;

    search_start_time = std::chrono::steady_clock::now();

    searching = true;
    // reset stop bool for min wait time
    min_work_bool = false;
    budget_reached = false;
    // reset the thread stopper
    thread_stopper = std::stop_source();

    Search::core_count = core_count;

    root_state = state;
    loadSearchOpponent();
    searchOpponent(root_state);
    searchGarbage(root_state);

    time = 0;

    // Create new search tree
    uct = UCT(core_count);
    parked_ids.assign(core_count, {});
    depth_baseline.assign(core_count, DepthBaseline{});

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
        EmulationGame start = root_state;
        startFromRoot(start);
        queues[rootOwnerIdx]->enqueue(std::make_unique<Job>(std::move(start), SELECT), core_count);
    }

    replay_root = root_state;
    replay_root_hash = root_state.hash();
    loadTuning();
    resetWorkerLocal(core_count);

    // Spawn worker threads
    for (auto& idx : core_indices) {
        worker_threads[idx] = std::jthread(worker(), thread_stopper.get_token(), idx);
    }

    initialised = true;
};

void Search::continueSearch(EmulationGame state) {
    //std::cout << "started searching, root hash is: " << (int)(state.hash() % 1000) << std::endl;

    if (state.game_over) {
        return;
    }

    // Collect here rather than in endSearch, so the move is returned first.
    if (!UCT::retain_for_dump && uct.map_size() > 200000) {
        uct.collect();
    }

    search_start_time = std::chrono::steady_clock::now();

    searching = true;
    // reset stop bool for min wait time
    min_work_bool = false;
    budget_reached = false;
    // reset the thread stopper
    thread_stopper = std::stop_source();

    root_state = state;

    time = time + 1;

    root_state.attack = state.app() * 0;
    root_state.true_attack = state.true_app() * 0;
    root_state.pieces = 0;
    root_state.lines = 2;
    loadSearchOpponent();
    searchOpponent(root_state);
    searchGarbage(root_state);
    depth_baseline.resize(core_count);
    for (DepthBaseline& baseline : depth_baseline) {
        baseline.shift();
    }

    if (!uct.nodeExists(state.hash())) {
        uct.insertNode(UCTNode(state));
    }
    else if (uct.getNode(state.hash()).actionCount() == 0) {
        UCTNode fresh(state);
        if (fresh.actionCount() > 0) {
            uct.getNode(state.hash()) = std::move(fresh);
        }
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
            EmulationGame start = root_state;
            startFromRoot(start);
            queues[rootOwnerIdx]->enqueue(std::make_unique<Job>(std::move(start), SELECT), core_count);
        }

    }

    replay_root = root_state;
    replay_root_hash = root_state.hash();
    loadTuning();
    resetWorkerLocal(core_count);

    for (auto& idx : core_indices) {
        worker_threads[idx] = std::jthread(worker(), thread_stopper.get_token(), idx);
    }
}

static size_t mostVisited(const UCTNode& root, const std::vector<float>* extra_N = nullptr,
                          const std::vector<float>* extra_R = nullptr) {
    float biggest_N = 0;
    float biggest_R = 0;
    size_t best = 0;
    for (size_t i = 0; i < root.actionCount(); i++) {
        float N = root.stats(i).N + (extra_N ? (*extra_N)[i] : 0);
        float R = root.stats(i).R + (extra_R ? (*extra_R)[i] : 0);
        if (N > biggest_N || (N == biggest_N && R > biggest_R)) {
            biggest_N = N;
            biggest_R = R;
            best = i;
        }
    }
    return best;
}

static void flushParked() {
    using Search::uct;
    const uint32_t root_hash = Search::root_state.hash();
    if (!uct.nodeExists(root_hash)) {
        return;
    }
    UCTNode& root = uct.getNode(root_hash);
    std::vector<float> hidden_N(root.actionCount(), 0.0f);
    std::vector<float> hidden_R(root.actionCount(), 0.0f);

    // Ids, not pointers: getNode below moves old-generation nodes, which would
    // leave a pointer to a parked node dangling.
    std::vector<uint32_t> parked;
    for (std::vector<uint32_t>& ids : parked_ids) {
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        for (const uint32_t id : ids) {
            const UCTNode* node = findNode(id);
            if (node && (node->N_buffer != 0 || !rewardEmpty(node->R_buffer))) {
                parked.push_back(id);
            }
        }
        ids.clear();
    }

    for (const uint32_t id : parked) {
        UCTNode* node = findNode(id);
        if (!node) {
            continue;
        }
        const float N = (float)node->N_buffer;
#if NANA_CC_SEARCH
        Reward R = node->R_buffer;
#else
        Reward R = node->value();
#endif
        UCTNode* child = node;
        for (size_t step = 0; step < MAX_PATH_DEPTH; step++) {
            const uint32_t parent_hash = child->parent_hash;
            const int action = child->parent_action;
            if (action < 0 || child->id == root_hash || !uct.nodeExists(parent_hash)) {
                break;
            }
            UCTNode& parent = uct.getNode(parent_hash);
            if ((size_t)action >= parent.actionCount()) {
                break;
            }
#if NANA_CC_SEARCH
            R = parent.climb(action, R, true);
#else
            R = parent.report(action, child->id, R, N, false);
#endif
            parent.stats(action).N += N;
            parent.N += (int)N;
            if (parent_hash == root_hash) {
                hidden_N[action] += N;
                hidden_R[action] += rewardScalar(R);
                break;
            }
            child = &parent;
        }
        if (UCTNode* parked_node = findNode(id)) {
            parked_node->N_buffer = 0;
            parked_node->R_buffer = Reward{};
        }
    }

    std::vector<float> minus_N(hidden_N.size()), minus_R(hidden_R.size());
    for (size_t i = 0; i < hidden_N.size(); i++) {
        minus_N[i] = -hidden_N[i];
        minus_R[i] = -hidden_R[i];
    }
    const size_t before = mostVisited(root, &minus_N, &minus_R);
    const size_t after = mostVisited(root);
    float total = 0, hidden = 0;
    for (size_t i = 0; i < root.actionCount(); i++) {
        total += root.stats(i).N;
        hidden += hidden_N[i];
    }
    Search::flush_stats.searches++;
    Search::flush_stats.changed += before != after;
    Search::flush_stats.hidden_share += total > 0 ? hidden / total : 0;
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
    EmulationGame::deterministic_chance = false;


    // reset stop bool for min wait time
    min_work_bool = false;
    budget_reached = false;
    // reset the thread stopper
    thread_stopper = std::stop_source();

    bool delivered = true;
    while (delivered) {
        delivered = false;
        for (int i = 0; i < core_count; i++) {
            while (!queues[i]->isempty()) {
                Msg msg = queues[i]->dequeue();
                if (msg->type == FLUSH) {
                    withOpponent([&]<class Opp>() { processJob<Opp>(i, std::move(msg)); });
                    delivered = true;
                }
            }
        }
    }

    flushParked();

    queues.clear();
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
    std::printf("thread    wait  select  backprop  expand   put | rollout |   jobs/s | ns/job: select backprop expand    put rollout | queue us   top\n");
    for (int i = 0; i < (int)uct.stats.size(); i++) {
        const WorkerStatistics& s = uct.stats[i];
        uint64_t total = s.wait_cycles;
        for (int t = 0; t < JOB_TYPE_COUNT; t++) total += s.job_cycles[t];
        if (total == 0) continue;
        uint64_t jobs = 0;
        for (int t = 0; t < JOB_TYPE_COUNT; t++) jobs += s.job_count[t];
        auto pct = [&](uint64_t c) { return 100.0 * c / total; };
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

static int myLoad(int threadIdx) {
    return (int)worker_local[threadIdx].pending.size() * tuning.rollout_weight
        + (int)Search::queues[threadIdx]->buffered() * tuning.message_weight;
}

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

static int chooseVictim(int chooser, int requester, uint64_t candidates) {
    const WorkerLocal& local = worker_local[chooser];
    const int n = Search::core_count;

    struct Candidate {
        float score;
        int thread;
    };
    Candidate top[MAX_STEAL_TOPK];
    int count = 0;

    int start = (int)(fast_seed() % (uint32_t)n);
    for (int step = 0; step < n; step++) {
        int j = (start + step) % n;
        if (!(candidates & (1ull << j))) continue;

        float score = (float)local.load_view[j];
        if (worker_l3_group[j] != worker_l3_group[requester]) {
            score -= (float)tuning.steal_chiplet_penalty;
        }

        int i;
        if (count < tuning.steal_topk) {
            i = count++;
        }
        else if (score > top[count - 1].score) {
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
        float u = ((float)(fast_seed() >> 8) + 0.5f) * (1.0f / 16777216.0f);
        float value = top[i].score / (float)tuning.steal_temperature - std::log(-std::log(u));
        if (value > best) {
            best = value;
            chosen = top[i].thread;
        }
    }
    return chosen;
}

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

static void handleRequest(int threadIdx, Msg&& msg) {
    Job& request = *msg;
    WorkerLocal& local = worker_local[threadIdx];
    int requester = request.requester;

    int requester_load = request.requester_load;
    if (!local.pending.empty() && myLoad(threadIdx) - tuning.rollout_weight >= requester_load + tuning.give_margin) {
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
        request.type = NO_WORK;
        send(threadIdx, requester, std::move(msg));
        return;
    }

    int next = chooseVictim(threadIdx, requester, unvisited);
    request.visited |= 1ull << next;
    Search::uct.stats[threadIdx].forwards++;
    send(threadIdx, next, std::move(msg));
}

static void notifySimulationDone() {
    if (!min_work_bool.load(std::memory_order_relaxed)) {
        // extra scope to prevent notifying the main thread while holding the lock
        // this is to prevent the main thread from waiting on the condition variable while we're holding the lock
        {
            std::scoped_lock lock(min_work_mutex);
            min_work_bool = true;
        }
        min_work_cv.notify_one();
    }
}

// A CC backup settled at the node it just left (job.child_hash), so every node above it keeps its
// values and sampling weights. Draw the next simulation's path from the root again: fresh chance, and at
// each ancestor the stored action stays with the probability it was sampled with. The job selects at the
// first node where the draw leaves the stored path (without the stored action, when an action draw left
// it), or at the settled node itself, which this thread owns.
template <class Opp>
[[maybe_unused]] static void restartCC(int threadIdx, Msg&& msg) {
    Job& job = *msg;
    UCT& uct = Search::uct;
    RNG& rng = uct.rng[threadIdx];
    const size_t depth = job.path.size();

    EmulationGame& state = job.state;
    state = replay_root;
    state.start_from_root<Opp>();
    size_t keep = depth;
    int exclude = -1;
    for (size_t i = 0; i < depth; i++) {
        if (ccUniform(rng) >= job.probs[i]) {
            keep = i;
            exclude = job.path[i].actionID;
            break;
        }
        state.set_move(state.specific_move(job.moves[i].unpack()));
        state.play_moves<Opp>();
        state.chance_move<Opp>();
        const uint32_t expected = i + 1 < depth ? job.path[i + 1].hash : job.child_hash;
        if (state.game_over || state.hash() != expected) {
            keep = i + 1;
            break;
        }
    }

    uct.stats[threadIdx].restarts++;
    const bool matched = keep == depth && exclude < 0 && !state.game_over && state.hash() == job.child_hash;
    if (matched) {
        uct.stats[threadIdx].replay_matched++;
        uct.stats[threadIdx].restart_depth += depth;
    }
    else {
        uct.stats[threadIdx].replay_diverged++;
        uct.stats[threadIdx].replay_diverge_depth += keep;
        uct.stats[threadIdx].restart_depth += keep;
        job.path.resize(keep, HashActionPair(0, -1));
        job.moves.resize(keep);
        job.probs.resize(keep);
    }

    job.exclude = exclude;
    job.R = Reward{};
    job.extra_N = 0;
    job.history.clear();
    job.steps.clear();
    job.snapshots.clear();
    job.type = SELECT;

    notifySimulationDone();

    send(threadIdx, matched ? threadIdx : uct.getOwner(state.hash()), std::move(msg));
}

template <class Opp>
void Search::processJob(const int threadIdx, Msg&& msg) {
    Job& job = *msg;

    if (job.type == FLUSH) {
        uct.stats[threadIdx].flushes++;
        UCTNode& node = uct.getNode(job.path.back().hash, threadIdx);
        EdgeStats& edge = node.stats(job.path.back().actionID);
#if !NANA_CC_SEARCH
        job.R = node.report(job.path.back().actionID, job.child_hash, job.R, 1.0f + job.extra_N, false);
#else
        job.R = node.climb(job.path.back().actionID, job.R, true);
#endif
        edge.N += job.extra_N;
        node.N += job.extra_N;
        if (job.path.size() == 1) {
            edge.N += 1;
            node.N += 1;
            if (node.id != replay_root_hash) {
                noteParked(threadIdx, node);
                if constexpr (cc_search) {
                    node.R_buffer += job.R;
                }
                node.N_buffer += job.extra_N + 1;
            }
            return;
        }
        job.child_hash = job.path.back().hash;
        job.R += node.R_buffer;
        job.extra_N += node.N_buffer;
        node.R_buffer = Reward{};
        node.N_buffer = 0;
        job.path.pop_back();
        send(threadIdx, uct.getOwner(job.path.back().hash), std::move(msg));
        return;
    }

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

        // the owner runs parked expansions only once its queue is empty, so another job parked for
        // the same state may have expanded it meanwhile: select into that node instead of expanding
        // it again (the depth cap's leaves are expanded on purpose, as SELECT does)
        if (const uint32_t hash = job.state.hash();
            uct.getOwner(hash) == (uint32_t)threadIdx && uct.nodeExists(hash) && job.path.size() < leafDepth()) {
            job.type = SELECT;
            send(threadIdx, threadIdx, std::move(msg));
            return;
        }

        uct.stats[threadIdx].nodes++;

        Reward reward = rollout<Opp>(job.state, threadIdx, job.path.size(), job.parent_board);

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
        job.child_hash = hash;

        if (state.game_over) {

            Reward reward = rollout<Opp>(state, threadIdx, job.path.size(), job.parent_board);

            if (job.path.empty()) {
                notifySimulationDone();
                return;
            }

            uint32_t parent_hash = job.path.back().hash;

            uint32_t parentIdx = uct.getOwner(parent_hash);

            job.R = reward;
            job.extra_N = 0;
            job.depth = state.pieces;
            job.type = BACKPROP;

            send(threadIdx, parentIdx, std::move(msg));

            return;
        }

        bool exists = uct.nodeExists(hash);
        if (exists && job.path.size() >= leafDepth()) {
            uct.stats[threadIdx].depth_capped++;
            exists = false;
        }

        if (exists) {

            UCTNode& node = uct.getNode(hash, threadIdx);

            if (!job.path.empty()) {
                node.parent_hash = job.path.back().hash;
                node.parent_action = job.path.back().actionID;
            }

            if (node.actionCount() == 0) {
                state.game_over = true;
                send(threadIdx, threadIdx, std::move(msg));
                return;
            }

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

            if (NANA_MPMCTS && node.N >= HISTORY_MIN_VISITS) {
                node.history = job.history;
            }

            constexpr bool single_pass = NANA_MPMCTS;
            if constexpr (single_pass) {
                action = node.selectWithCompetitors(depth, row);
            }
            else if constexpr (cc_search) {
                float p = 1.0f;
                action = node.select_CC(uct.rng[threadIdx], job.exclude, &p);
                job.exclude = -1;
                if constexpr (cc_restart) {
                    job.probs.push_back(p);
                }
            }
            else {
                action = node.select(depth);
            }

            // Virtual Loss by setting N := N+1
            node.N += 1;

            EdgeStats& edge = node.stats(action);
            edge.addN();

            node.updateTime(action, time);

            if constexpr (NANA_MPMCTS) {
                if constexpr (single_pass) {
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

            {
                const PieceType type = node.move(action).piece.type;
                const PieceType hold_candidate = state.game.hold ? *state.game.hold : state.game.queue.front();
                if (type != state.game.current_piece.type && type != hold_candidate) {
                    uct.stats[threadIdx].illegal_moves++;
                }
            }
            job.moves.push_back(node.cold[action].move);
            job.parent_board = node.cold[action].eval - node.cold[action].move_reward;
            state.set_move(state.specific_move(node.move(action)));

            [[maybe_unused]] const int attack_before = state.true_attack;
            state.play_moves<Opp>();
#if !NANA_CC_SEARCH
            {
                const float step = (float)(state.true_attack - attack_before);
                node.cold[action].step_attack = step;
                job.steps.push_back(step);
            }
#endif
            state.chance_move<Opp>();

            uint32_t new_hash = state.hash();
            job.child_hash = new_hash;

            // Get the owner of the updated state based on the hash
            uint32_t ownerIdx = uct.getOwner(new_hash);

            job.path.push_back(HashActionPair(hash, (int)action));

            uct.stats[threadIdx].deepest_node = std::max(uct.stats[threadIdx].deepest_node, (uint64_t)job.path.size());

            send(threadIdx, ownerIdx, std::move(msg));
        }
        else {
            uct.stats[threadIdx].deferred++;
            job.type = EXPAND;
            worker_local[threadIdx].pending.push_back(std::move(msg));
        }
    }
    else if (job.type == BACKPROP) {

        uct.stats[threadIdx].backprop_messages++;
        UCTNode& node = uct.getNode(job.path.back().hash, threadIdx);
        EdgeStats& edge = node.stats(job.path.back().actionID);

        Reward reward = job.R;
        [[maybe_unused]] bool settled = false;
#if !NANA_CC_SEARCH
        reward = node.report(job.path.back().actionID, job.child_hash, job.R, 1.0f + job.extra_N, true);
#else
        const float before = node.ccValue();
        edge.R = reward <= CC_DEAD ? CC_DEAD : node.cold[job.path.back().actionID].move_reward + reward;
        reward = node.ccValue();
        settled = reward == before;
#endif

        edge.N += job.extra_N;
        node.N += job.extra_N;

        job.child_hash = job.path.back().hash;
        job.path.pop_back();
        if (!job.moves.empty()) {
            job.moves.pop_back();
        }
        if (!job.steps.empty()) {
            job.steps.pop_back();
        }
        if (!job.probs.empty()) {
            job.probs.pop_back();
        }

        if constexpr (cc_restart) {
            if (settled && !job.path.empty() && job.moves.size() == job.path.size() &&
                job.probs.size() == job.path.size()) {
                restartCC<Opp>(threadIdx, std::move(msg));
                return;
            }
        }

        if (job.path.empty() || settled) {
            if (job.path.empty()) {
                uct.stats[threadIdx].root_backprops++;
            }
            job.path.clear();

            job.state = root_state;
            job.state.start_from_root<Opp>();
            job.R = Reward{};
            job.extra_N = 0;
            job.history.clear();
            job.moves.clear();
            job.steps.clear();
            job.snapshots.clear();
            job.probs.clear();
            job.exclude = -1;
            job.type = SELECT;

            notifySimulationDone();

            if constexpr (cc_search) {
                send(threadIdx, uct.getOwner(root_state.hash()), std::move(msg));
            }
            else {
                queues[threadIdx]->enqueue(std::move(msg), threadIdx);
            }

            return;
        }

        if constexpr (!NANA_MPMCTS) {
            job.R = reward;
            send(threadIdx, uct.getOwner(job.path.back().hash), std::move(msg));
            return;
        }

        job.history.pop_back();
        EmulationGame here = std::move(job.snapshots.back());
        job.snapshots.pop_back();

        static thread_local std::vector<HistoryRow> rows;
        rows = job.history;
        for (size_t i = 0; i < rows.size() && i < node.history.size(); i++) {
            const HistoryRow& kept = node.history[i];
            if (kept.hash == rows[i].hash && kept.chosen == rows[i].chosen && kept.node_N > rows[i].node_N) {
                rows[i] = kept;
            }
        }

        bool still_chosen = true;
#if !NANA_CC_SEARCH
        {
            Return v = reward;
            for (size_t j = rows.size(); j-- > 0;) {
                v = stepUp(v, j < job.steps.size() ? job.steps[j] : 0.0f);
                rows[j].edges[0].R += rewardScalar(v);
            }
        }
#endif
        for (HistoryRow& row : rows) {
            row.edges[0].N += job.extra_N;
            row.node_N += job.extra_N;
            still_chosen = still_chosen && row.stillChosen();
        }

        if (still_chosen) {
            noteParked(threadIdx, node);
            node.N_buffer += job.extra_N + 1;

            for (HistoryRow& row : rows) {
                row.edges[0].N += 1;
                row.node_N += 1;
            }

            uct.stats[threadIdx].restarts++;
            uct.stats[threadIdx].restart_depth += job.path.size();

            notifySimulationDone();

            if (node.N >= HISTORY_MIN_VISITS) {
                node.history = rows;
            }

            const size_t depth = job.path.size();
            size_t left = depth + 1;
            if (job.moves.size() == depth) {
                left = replayPath<Opp>(job, node.id, job.state);
            }
            else {
                job.state = std::move(here);
            }
            if (left <= depth) {
                node.N_buffer -= job.extra_N + 1;
                uct.stats[threadIdx].restarts--;
                uct.stats[threadIdx].restart_depth -= depth;
                uct.stats[threadIdx].replay_diverged++;
                uct.stats[threadIdx].replay_diverge_depth += left;

                Msg flush = std::make_unique<Job>(FLUSH);
                flush->R = reward + node.R_buffer;
                flush->extra_N = job.extra_N + node.N_buffer;
                flush->child_hash = job.child_hash;
                node.R_buffer = Reward{};
                node.N_buffer = 0;
                flush->path.assign(job.path.begin() + (left - 1), job.path.end());
                send(threadIdx, uct.getOwner(flush->path.back().hash), std::move(flush));

                job.path.erase(job.path.begin() + left, job.path.end());
                job.moves.erase(job.moves.begin() + left, job.moves.end());
                job.steps.erase(job.steps.begin() + std::min(left, job.steps.size()), job.steps.end());
                job.snapshots.erase(job.snapshots.begin() + std::min(left, job.snapshots.size()), job.snapshots.end());
                rows.erase(rows.begin() + std::min(left, rows.size()), rows.end());
                job.history = rows;
                job.R = Reward{};
                job.extra_N = 0;
                job.type = SELECT;
                send(threadIdx, uct.getOwner(job.state.hash()), std::move(msg));
                return;
            }
            uct.stats[threadIdx].replay_matched++;
            job.history = rows;
            job.R = Reward{};
            job.extra_N = 0;
            job.type = SELECT;
            send(threadIdx, threadIdx, std::move(msg));
            return;
        }

        if (node.N >= HISTORY_MIN_VISITS) {
            node.history = rows;
        }

        job.R = reward + node.R_buffer;
        job.extra_N += node.N_buffer;
        node.R_buffer = Reward{};
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

template <class Opp>
void Search::search(std::stop_token stop, const int threadIdx) {
    pinThreadToCore(threadIdx);

#ifdef NANA_BENCH
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
        processJob<Opp>(threadIdx, std::move(msg));
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

        if (simulation_limit != 0 && simulationsSoFar() >= simulation_limit) {
            {
                std::scoped_lock lock(min_work_mutex);
                min_work_bool = true;
            }
            min_work_cv.notify_one();
            {
                std::scoped_lock lock(budget_mutex);
                budget_reached = true;
            }
            budget_cv.notify_all();
            return;
        }

        uint64_t t0 = 0;
#ifdef NANA_BENCH
        t0 = __rdtsc();
#endif
        bool has_message = !queues[threadIdx]->isempty();

        maybeRequest(threadIdx);

        if (has_message) {
            Msg msg = queues[threadIdx]->dequeue();
            Job& job = *msg;
            if (job.sender < core_count) {
                local.load_view[job.sender] = job.sender_load;
            }
#ifdef NANA_BENCH
            if (job.sent_at) {
                int64_t queued = (int64_t)(__rdtsc() - job.sent_at);
                profile.local.queue_cycles += queued > 0 ? queued : 0;
                profile.local.queue_count++;
                job.sent_at = 0;
            }
#endif
            if (job.type == STOP) {
                return;
            }
            run(std::move(msg), t0);
        }
        else if (!local.pending.empty()) {
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


template <class Opp>
Reward Search::rollout(EmulationGame& state, int threadIdx, [[maybe_unused]] size_t depth,
                       [[maybe_unused]] float parent_board) {
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

    uct.stats[threadIdx].nodes++;
    uct.stats[threadIdx].rollouts++;


    [[maybe_unused]] const Return dead{ 0, 0, 0, 1, 1 };

    if (state.game_over) {
#if NANA_CC_SEARCH
        return CC_DEAD;
#else
        return dead;
#endif
    }

    UCTNode node(state);

    float max_eval = 0.0;

    for (size_t i = 0; i < node.actionCount(); i++) {
        max_eval = std::max(max_eval, node.eval(i));
    }

    const bool stuck = node.actionCount() == 0;

#if NANA_BELIEF_LOG
    if (!stuck) {
        logBelief(state, depth, max_eval - parent_board);
    }
#endif

#if NANA_CC_SEARCH
    float value = CC_DEAD;
    for (size_t i = 0; i < node.actionCount(); i++) {
        value = std::max(value, node.eval(i));
    }
    maybeInsertNode(std::move(node), threadIdx);
    return value;
#else
    if (!stuck) {
        max_eval = depth_baseline[threadIdx].correct(depth, max_eval, parent_board);
    }
    const Reward value = stuck ? dead : Return{ 0, 0, max_eval, 1, 0 };
    node.leaf_V = value;
    maybeInsertNode(std::move(node), threadIdx);
    return value;
#endif
}

Move Search::bestMove() {

    int biggest_N = 0;
    float biggest_R = 0;
    Move best_move;

    UCTNode& root = uct.getNode(root_state.hash());
    for (size_t i = 0; i < root.actionCount(); i++) {
        const EdgeStats& action = root.stats(i);
        if constexpr (cc_search) {
            if (best_move.null_move || action.R > biggest_R) {
                biggest_R = action.R;
                best_move = root.move(i);
            }
            continue;
        }
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
    }

    return best_move;
}
