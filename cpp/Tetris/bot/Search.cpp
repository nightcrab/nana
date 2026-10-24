#include "Search.hpp"
#include "Config.hpp"
#include <cstring>
#include <array>

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

EmulationGame Search::root_state(false);

// stuff for notifying the main thread that the worker threads have at least a minimum amount of work done
std::mutex min_work_mutex;
std::condition_variable min_work_cv;
std::atomic_bool min_work_bool;

uint64_t Search::simulation_limit = 0;
const RootOpponent* Search::root_opponent = nullptr;
const OpponentFilter* Search::root_filter = nullptr;
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

// jobs in flight per worker
#ifndef NANA_LOAD_FACTOR
#define NANA_LOAD_FACTOR 6
#endif
constexpr int LOAD_FACTOR = NANA_LOAD_FACTOR;

// selects past this depth are treated as leaves; a safety net for cycles in the state graph
constexpr size_t MAX_PATH_DEPTH = 64;
// nodes this many moves below the root are leaves: config depth_cap, or NANA_DEPTH_CAP for builds
// that fix it (strength variants), or the environment variable NANA_DEPTH_CAP in builds that don't
// (measurement: one binary for a sweep); 0 = no cap
#ifndef NANA_DEPTH_CAP
#define NANA_DEPTH_CAP 0
#endif
static int configuredDepthCap() {
    if (NANA_DEPTH_CAP > 0) return NANA_DEPTH_CAP;
    if (const char* text = std::getenv("NANA_DEPTH_CAP"); text && *text) return std::atoi(text);
    return config().depth_cap;
}
static size_t leafDepth() {
    static const size_t depth = [] {
        const int cap = configuredDepthCap();
        return cap > 0 && (size_t)cap < MAX_PATH_DEPTH ? (size_t)cap : MAX_PATH_DEPTH;
    }();
    return depth;
}


static_assert(!NANA_SUFFIX || Search::search_style == NANA, "NANA_SUFFIX is for the PUCT (NANA) search");

// 1 = MP-MCTS: backprops stop and restart selection where every ancestor would choose the same path.
// 0 = TDS-UCT: every backprop goes to the root
#ifndef NANA_MPMCTS
#define NANA_MPMCTS 1
#endif

// leaky gate for MP-MCTS restarts (notes/leaky-gate-mp-mcts.md): a leak carries a backprop on up
// although every ancestor would still choose its path, delivering what's parked below them.
// when a would-be restart leaks:
// 0 = never
// 1 = with probability NANA_LEAK_P
// 2 = once the node's parked visits would reach NANA_LEAK_K
// 3 = once they'd reach NANA_LEAK_EPS x the node's visits (0 = always: every backprop reaches the root)
#ifndef NANA_LEAK
#define NANA_LEAK 0
#endif
#ifndef NANA_LEAK_P
#define NANA_LEAK_P 0.05
#endif
#ifndef NANA_LEAK_K
#define NANA_LEAK_K 32
#endif
#ifndef NANA_LEAK_EPS
#define NANA_LEAK_EPS 0.05
#endif

// a restart at node X parks its reward for X's parent edge, and the root's edge to its favoured
// child only gets the parked results when the root's choice moves away from that child. so when a
// search ends, the root's statistics (which bestMove reads) can miss much of the search, and tree
// reuse starts the next search from them. like TDS-df-UCT forcing every job back to the root when
// its search ends (Yoshizoe et al. 2011), at the end of each search (workers stopped):
// 0 = leave the buffers parked (the original behaviour)
// 1 = walk every parked buffer up its parent chain to the root first
// 2 = only measure what that would change (Search::flush_stats), leave the buffers parked
#ifndef NANA_FLUSH_AT_END
#define NANA_FLUSH_AT_END 1
#endif

// restarts in a stochastic game: "every ancestor would choose the same edge" only means a fresh
// select would take the same moves. the chance between them (the next piece past the preview, the
// opponent's garbage, both in the hash) could still take it to other nodes, and the state a restart
// resumes from (the job's snapshot) is one sampled future: the hidden queue, the opponent's stack.
// so on restart:
// 1 = replay the path's moves from the search root with fresh chance, as a fresh select would. if it
//     reaches the restart node, restart there with the replayed state; if it leaves the path at
//     depth k, send a SELECT with the replayed state to the node it reached, and a FLUSH that carries
//     this result and the abandoned buffers up to depth k - 1. a node's moves must then be legal in
//     any state reaching it: NANA_HASH_QUEUE_FRONT
// 0 = resume from the snapshot (the original behaviour)
#ifndef NANA_REPLAY_RESTART
#define NANA_REPLAY_RESTART 1
#endif
static_assert(!NANA_REPLAY_RESTART || NANA_HASH_QUEUE_FRONT, "NANA_REPLAY_RESTART needs NANA_HASH_QUEUE_FRONT");

// the search root as the workers start (root_state's rng is reseeded by workers on root backprops)
static EmulationGame replay_root(false);
static uint32_t replay_root_hash = 0;

// play the job's path from the search root with fresh chance. returns the depth of the first node
// the replay doesn't reach (path.size() + 1 if it reaches restart_hash after the whole path), with
// the replayed state at that depth in state
[[maybe_unused]] static size_t replayPath(const Job& job, uint32_t restart_hash, EmulationGame& state) {
    state = replay_root;
    state.start_from_root();
    const size_t depth = job.path.size();
    for (size_t i = 0; i < depth; i++) {
        state.set_move(state.specific_move(job.moves[i].unpack()));
        state.play_moves();
        state.chance_move();
        const uint32_t expected = i + 1 < depth ? job.path[i + 1].hash : restart_hash;
        if (state.game_over || state.hash() != expected) {
            return i + 1;
        }
    }
    return depth + 1;
}

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
    if (const auto* c = tbpConfig()) {
        tuning.message_weight = c->message_weight;
        tuning.rollout_weight = c->rollout_weight;
        tuning.low_water = c->low_water;
        tuning.give_margin = c->give_margin;
        tuning.backoff_iterations = c->backoff_iterations;
        tuning.steal_chiplet_penalty = c->steal_chiplet_penalty;
        tuning.steal_topk = c->steal_topk;
        tuning.steal_temperature = c->steal_temperature;
        return;
    }
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

// what a rollout is worth. kill (the default): with r our own score (APP and eval, squashed into
// [0, 1) by x / (x + scale)) and K the expected discounted kill of the opponent (along the path,
// then the HMM's death_value from the leaf), the value is K' + (1 - K') r with K' = (1 - lambda) K:
// a kill is worth 1, less lambda of the way back to r. app: APP and eval only, the opponent's death
// ignored, for the APP strength benchmarks. set from the environment when a search starts:
// NANA_REWARD=kill|app, NANA_REWARD_SCALE, NANA_KILL_LAMBDA, NANA_KILL_DISCOUNT (only when the HMM
// has no fitted discount)
struct RewardTuning {
    bool app_only = false;
    float scale = 0.25f;
    float lambda = 0.0f;
    float discount = 0.9f;
};

static RewardTuning reward_tuning;

static void loadRewardTuning() {
    if (const auto* c = tbpConfig()) {
        reward_tuning.app_only = c->reward == "app";
        reward_tuning.scale = static_cast<float>(c->reward_scale);
        reward_tuning.lambda = static_cast<float>(c->kill_lambda);
        reward_tuning.discount = static_cast<float>(c->kill_discount);
        return;
    }
    reward_tuning = RewardTuning();
    if (const char* text = std::getenv("NANA_REWARD")) {
        reward_tuning.app_only = std::strcmp(text, "app") == 0;
    }
    auto read = [](const char* name, float& value) {
        // an empty variable counts as unset (atof would read it as 0)
        if (const char* text = std::getenv(name); text && *text) {
            value = (float)std::atof(text);
        }
    };
    read("NANA_REWARD_SCALE", reward_tuning.scale);
    read("NANA_KILL_LAMBDA", reward_tuning.lambda);
    read("NANA_KILL_DISCOUNT", reward_tuning.discount);
}

// carry the HMM opponent on from what the caller knows of the real one, if anything, and publish the
// belief for the simulations to sample from; otherwise the opponent starts fresh
static void adoptOpponentBelief(EmulationGame& root) {
    markov::root_belief.active = false;
    if (Search::root_opponent) {
        root.markov.adopt(*Search::root_opponent);
        Search::root_opponent->publish();
    }
}

// count the opponent's death from the search root, in the discount the reward uses
static void resetKill(EmulationGame& root) {
    loadRewardTuning();
    const float fitted = markov::params().discount;
    const float discount = fitted > 0 ? fitted : reward_tuning.discount;
    root.reset_kill(discount, !reward_tuning.app_only);
    if (NANA_SUFFIX && !reward_tuning.app_only) {
        std::cerr << "NANA_SUFFIX needs NANA_REWARD=app (reward \"app\" in nana.json)" << std::endl;
        std::exit(2);
    }
}

// NANA_SEARCH_GARBAGE=0 (measurement): the opponent sends nothing in the search;
// garbage already on the meter still comes.
#ifndef NANA_SEARCH_GARBAGE
#define NANA_SEARCH_GARBAGE 1
#endif
// NANA_PHANTOM_GARBAGE=1 with NANA_SEARCH_GARBAGE=0 (measurement): the attacks are still drawn and
// hashed through a meter that never lands (EmulationGame::phantom_meter), so the tree splits as with
// search garbage while the boards and values stay as without
#ifndef NANA_PHANTOM_GARBAGE
#define NANA_PHANTOM_GARBAGE 0
#endif
static void searchGarbage(EmulationGame& root) {
    if constexpr (NANA_FUTURE_ORACLE) {
        EmulationGame::deterministic_chance = true;
    }
    if constexpr (!NANA_SEARCH_GARBAGE) {
        // the sampled opponent's attacks scaled to nothing
        root.garbage_scale = 0;
        root.phantom_garbage = NANA_PHANTOM_GARBAGE;
        root.phantom_meter.clear();
    }
}

// NANA_OPPONENT_ORACLE (measurement only, emulation): the search's opponent is the real game's,
// hidden state and all, instead of the model's fresh one. simulations still draw fresh randomness, so
// they sample its futures without the model's bias
#ifndef NANA_OPPONENT_ORACLE
#define NANA_OPPONENT_ORACLE 0
#endif
static void searchOpponent(EmulationGame& root) {
    if constexpr (NANA_OPPONENT_ORACLE || NANA_FUTURE_ORACLE) {
        markov::root_belief.active = false;
    }
    else {
        // the game's own opponent is the HMM, true state and all: the search starts from its model
        root.reset_opponent();
        adoptOpponentBelief(root);
    }
    root.track_belief = Search::root_filter != nullptr;
    if (root.track_belief) {
        root.belief = *Search::root_filter;
    }
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

// NANA_DEPTH_BASELINE: some of what a leaf scores comes with time, not with the moves that got
// there (garbage landing, a bad piece), so lines look worse the further they're expanded. a leaf
// scores its eval less the mean step, at each depth down to it, from what the parent thought of the
// board its move left to what the child scores: the time-driven part averages out, and each line
// still pays for doing worse than the average step. paired (child against its own parent), so which
// lines the search chose to deepen doesn't leak in. each worker keeps its own; the next search
// shifts them up a ply and halves their weight
#ifndef NANA_DEPTH_BASELINE
#define NANA_DEPTH_BASELINE 1
#endif
struct alignas(64) DepthBaseline {
    std::array<double, MAX_PATH_DEPTH + 1> sum{};
    std::array<double, MAX_PATH_DEPTH + 1> count{};

    float mean(size_t d) const { return count[d] > 0 ? (float)(sum[d] / count[d]) : 0.0f; }

    // record a leaf at depth d scoring raw, and return what it scores instead
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

    // the root moved a piece down: depth d+1 of the last search is depth d of this one
    void shift() {
        for (size_t d = 0; d < MAX_PATH_DEPTH; d++) {
            sum[d] = sum[d + 1] * 0.5;
            count[d] = count[d + 1] * 0.5;
        }
        sum[MAX_PATH_DEPTH] = count[MAX_PATH_DEPTH] = 0;
    }
};
static std::vector<DepthBaseline> depth_baseline;

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
    budget_reached = false;
    // reset the thread stopper
    thread_stopper = std::stop_source();

    Search::core_count = core_count;

    root_state = state;
    // the game's own opponent is the HMM, true state and all: the search starts from its model
    searchOpponent(root_state);
    resetKill(root_state);
    searchGarbage(root_state);

    time = 0;

    // Create new search tree
    uct = UCT(core_count);
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
        start.start_from_root();
        queues[rootOwnerIdx]->enqueue(std::make_unique<Job>(std::move(start), SELECT), core_count);
    }

    replay_root = root_state;
    replay_root_hash = root_state.hash();
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
    budget_reached = false;
    // reset the thread stopper
    thread_stopper = std::stop_source();

    root_state = state;

    time = time + 1;

    root_state.attack = state.app() * 0;
    root_state.true_attack = state.true_app() * 0;
    root_state.pieces = 0;
    root_state.lines = 2;
    searchOpponent(root_state);
    resetKill(root_state);
    searchGarbage(root_state);

    depth_baseline.resize(core_count);
    for (DepthBaseline& baseline : depth_baseline) {
        baseline.shift();
    }

    if (!uct.nodeExists(state.hash())) {
        uct.insertNode(UCTNode(state));
    }
    else if (uct.getNode(state.hash()).actionCount() == 0) {
        // node hashes ignore garbage rows, so the tree may hold a node for another state with this
        // hash that had no legal placement. kept as the root, the search would think we're lost and
        // its jobs would end at the root: the real root's moves replace it
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
            start.start_from_root();
            queues[rootOwnerIdx]->enqueue(std::make_unique<Job>(std::move(start), SELECT), core_count);
        }

    }

    replay_root = root_state;
    replay_root_hash = root_state.hash();
    loadTuning();
    resetWorkerLocal(core_count);

    for (auto& idx : core_indices) {
        worker_threads[idx] = std::jthread(search, thread_stopper.get_token(), idx);
    }
}

// the root edge bestMove would pick, with optional extra visits and rewards per edge
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

// NANA_FLUSH_AT_END: every parked buffer belongs to all edges from its node's parent up to the
// root. walk each one up the recorded parent chain, recording what reaches the root's edges and,
// with apply, adding it to every edge on the way (as the backprops that never came would have).
// only called with the workers stopped, so it may touch every shard
static void flushParked(bool apply) {
    using Search::uct;
    const uint32_t root_hash = Search::root_state.hash();
    if (!uct.nodeExists(root_hash)) {
        return;
    }
    UCTNode& root = uct.getNode(root_hash);
    std::vector<float> hidden_N(root.actionCount(), 0.0f);
    std::vector<float> hidden_R(root.actionCount(), 0.0f);

    std::vector<UCTNode*> parked;
    for (auto* side : { &uct.nodes_left, &uct.nodes_right }) {
        for (auto& shard : *side) {
            for (auto& [id, node] : shard.obj_) {
                if (node.N_buffer != 0 || !rewardEmpty(node.R_buffer)) {
                    parked.push_back(&node);
                }
            }
        }
    }

    for (UCTNode* node : parked) {
        const float N = (float)node->N_buffer;
        Reward R = node->R_buffer;
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
            R = parent.climb(action, R, true, apply);
            if (apply) {
                parent.stats(action).N += N;
                parent.N += (int)N;
            }
            if (parent_hash == root_hash) {
                hidden_N[action] += N;
                hidden_R[action] += rewardScalar(R);
                break;
            }
            child = &parent;
        }
        if (apply) {
            node->N_buffer = 0;
            node->R_buffer = Reward{};
        }
    }

    // compare with the root as it was (with apply, the root already has the hidden visits)
    std::vector<float> minus_N(hidden_N.size()), minus_R(hidden_R.size());
    for (size_t i = 0; i < hidden_N.size(); i++) {
        minus_N[i] = -hidden_N[i];
        minus_R[i] = -hidden_R[i];
    }
    const size_t before = apply ? mostVisited(root, &minus_N, &minus_R) : mostVisited(root);
    const size_t after = apply ? mostVisited(root) : mostVisited(root, &hidden_N, &hidden_R);
    float total = 0, hidden = 0;
    for (size_t i = 0; i < root.actionCount(); i++) {
        total += root.stats(i).N + (apply ? 0 : hidden_N[i]);
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
    // Real play resumes with its own chance.
    EmulationGame::deterministic_chance = false;

    // reset stop bool for min wait time
    min_work_bool = false;
    budget_reached = false;
    // reset the thread stopper
    thread_stopper = std::stop_source();

    if constexpr (NANA_REPLAY_RESTART) {
        // FLUSH messages still queued hold results that are in no buffer yet: deliver them first
        // (workers are joined, so this thread may consume every queue)
        bool delivered = true;
        while (delivered) {
            delivered = false;
            for (int i = 0; i < core_count; i++) {
                while (!queues[i]->isempty()) {
                    Msg msg = queues[i]->dequeue();
                    if (msg->type == FLUSH) {
                        processJob(i, std::move(msg));
                        delivered = true;
                    }
                }
            }
        }
    }

    if constexpr (NANA_FLUSH_AT_END != 0) {
        flushParked(NANA_FLUSH_AT_END == 1);
    }

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
        sum.leaks += s.leaks;
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
    std::cout << "leaks (NANA_LEAK " << NANA_LEAK << "): " << sum.leaks << std::endl;
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

// tell the main thread a simulation has completed; it only needs telling once per search,
// so later simulations skip the mutex entirely
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

void Search::processJob(const int threadIdx, Msg&& msg) {
    // the job lives on the heap and belongs to whoever holds msg: don't touch it after sending
    Job& job = *msg;

    if (job.type == FLUSH) {
        // path.back() is the node whose edge gets the carried result; the last entry is the top
        uct.stats[threadIdx].flushes++;
        UCTNode& node = uct.getNode(job.path.back().hash, threadIdx);
        EdgeStats& edge = node.stats(job.path.back().actionID);
        job.R = node.climb(job.path.back().actionID, job.R, true);
        edge.N += job.extra_N;
        node.N += job.extra_N;
        if (job.path.size() == 1) {
            // the top: the diverged select's virtual visit goes through this edge too, and the
            // ancestors above still owe all of it, so park it here
            edge.N += 1;
            node.N += 1;
            if (node.id != replay_root_hash) {
                node.R_buffer += job.R;
                node.N_buffer += job.extra_N + 1;
            }
            return;
        }
        // this node is being left too: its own parked results belong to the edge above
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

        Reward reward = rollout(job.state, threadIdx, job.path.size(), job.parent_board);

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

            Reward reward = rollout(state, threadIdx, job.path.size(), job.parent_board);

            if (job.path.empty()) {
                // the root itself is lost: nothing to search, but don't leave endSearch waiting
                notifySimulationDone();
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
        if (exists && job.path.size() >= leafDepth()) {
            // safety net for cycles: treat it as a leaf
            uct.stats[threadIdx].depth_capped++;
            exists = false;
        }

        if (exists) {

            UCTNode& node = uct.getNode(hash, threadIdx);

            if constexpr (NANA_FLUSH_AT_END != 0) {
                if (!job.path.empty()) {
                    node.parent_hash = job.path.back().hash;
                    node.parent_action = job.path.back().actionID;
                }
            }

            if (node.actionCount() == 0) {
                // no legal placement: lost, like topping out. run it again as a terminal state
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

            {
                const PieceType type = node.move(action).piece.type;
                const PieceType hold_candidate = state.game.hold ? *state.game.hold : state.game.queue.front();
                if (type != state.game.current_piece.type && type != hold_candidate) {
                    uct.stats[threadIdx].illegal_moves++;
                }
            }
            if constexpr (NANA_REPLAY_RESTART) {
                job.moves.push_back(node.cold[action].move);
            }
            job.parent_board = node.cold[action].eval - node.cold[action].move_reward;
            state.set_move(state.specific_move(node.move(action)));

            [[maybe_unused]] const int attack_before = state.true_attack;
            state.play_moves();
#if NANA_SUFFIX
            {
                const float step = (float)(state.true_attack - attack_before);
                node.cold[action].step_attack = step;
                job.steps.push_back(step);
            }
#endif
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

        // Undo Virtual Loss by adding R. from here on, reward is what goes to the parent's edge
        Reward reward = job.R;
#if NANA_SUFFIX
        reward = node.climb(job.path.back().actionID, reward, false);
#else
        if constexpr (search_style == NANA) {
            reward = node.climb(job.path.back().actionID, reward, false);
        }
        if constexpr (search_style == CC) {
            if (reward > edge.R) {
                edge.R = reward;
            }
        }
#endif

        // simulations that restarted below us weren't counted on the way down
        edge.N += job.extra_N;
        node.N += job.extra_N;

        job.path.pop_back();
        if (!job.moves.empty()) {
            job.moves.pop_back();
        }
        if (!job.steps.empty()) {
            job.steps.pop_back();
        }

        if (job.path.empty()) {
            uct.stats[threadIdx].root_backprops++;

            // reuse this job for a fresh select from the root;
            // copy-assigning keeps the existing vector capacity. the copy gets fresh randomness and
            // its own draw of the opponent: workers never write the shared root_state
            job.state = root_state;
            job.state.start_from_root();
            job.R = Reward{};
            job.extra_N = 0;
            job.history.clear();
            job.moves.clear();
            job.steps.clear();
            job.snapshots.clear();
            job.type = SELECT;

            notifySimulationDone();

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
#if NANA_SUFFIX
        // each ancestor's edge would get the reward plus the moves between, whose steps are on the job
        {
            Return v = reward;
            for (size_t j = rows.size(); j-- > 0;) {
                constexpr float gamma = (float)NANA_SUFFIX_DISCOUNT;
                v.A = gamma * v.A + (v.n - v.n_dead) * (j < job.steps.size() ? job.steps[j] : 0.0f);
                v.L = gamma * v.L + (v.n - v.n_dead);
                rows[j].edges[0].R += rewardScalar(v);
            }
        }
        for (HistoryRow& row : rows) {
            row.edges[0].N += job.extra_N;
            row.node_N += job.extra_N;
            still_chosen = still_chosen && row.stillChosen();
        }
#else
        for (HistoryRow& row : rows) {
            row.edges[0].R += reward;
            row.edges[0].N += job.extra_N;
            row.node_N += job.extra_N;
            still_chosen = still_chosen && row.stillChosen();
        }
#endif

        // visits this node would have parked, counting this one, if it restarted here
        [[maybe_unused]] const int parked = node.N_buffer + job.extra_N + 1;
        bool leak = false;
        if constexpr (NANA_LEAK == 1) {
            // getRand(n) is (16-bit draw * n) >> 16 in 32 bits, so n must stay at or below 2^16
            constexpr uint32_t scale = 1u << 16;
            leak = still_chosen && uct.rng[threadIdx].getRand(scale) < (uint32_t)(NANA_LEAK_P * scale);
        }
        else if constexpr (NANA_LEAK == 2) {
            leak = still_chosen && parked >= NANA_LEAK_K;
        }
        else if constexpr (NANA_LEAK == 3) {
            leak = still_chosen && parked >= NANA_LEAK_EPS * node.N;
        }
        if (leak) {
            uct.stats[threadIdx].leaks++;
        }

        if (still_chosen && !leak) {
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

            // a simulation finished here too. when every ancestor keeps its choice (forced
            // moves, or all alternatives lost) no backprop may reach the root for the rest of
            // the search, and endSearch would wait forever
            notifySimulationDone();

            if (node.N >= HISTORY_MIN_VISITS) {
                node.history = rows;
            }

            if constexpr (NANA_REPLAY_RESTART) {
                const size_t depth = job.path.size();
                size_t left = depth + 1;
                if (job.moves.size() == depth) {
                    left = replayPath(job, node.id, job.state);
                }
                else {
                    // no move record to replay (shouldn't happen): restart from the snapshot
                    job.state = std::move(here);
                }
                if (left <= depth) {
                    // a fresh select would have left the path at depth `left`: undo the restart's
                    // parking here and send this result, with what's parked, up to depth left - 1
                    node.R_buffer -= reward;
                    node.N_buffer -= job.extra_N + 1;
                    uct.stats[threadIdx].restarts--;
                    uct.stats[threadIdx].restart_depth -= depth;
                    uct.stats[threadIdx].replay_diverged++;
                    uct.stats[threadIdx].replay_diverge_depth += left;

                    Msg flush = std::make_unique<Job>(FLUSH);
                    flush->R = reward + node.R_buffer;
                    flush->extra_N = job.extra_N + node.N_buffer;
                    node.R_buffer = Reward{};
                    node.N_buffer = 0;
                    flush->path.assign(job.path.begin() + (left - 1), job.path.end());
                    send(threadIdx, uct.getOwner(flush->path.back().hash), std::move(flush));

                    // the select carries on from the node the replay reached, keeping the
                    // path above it (the rows already have its visit through those depths)
                    // (erase, not resize: shrinking with resize needs default-constructible elements)
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

            here.reseed_chance();

            job.state = std::move(here);
            job.history = rows;
            job.R = Reward{};
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

        if (simulation_limit != 0 && simulationsSoFar() >= simulation_limit) {
            // endSearch waits for a finished simulation, which a tiny budget may not have yet
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


Reward Search::rollout(EmulationGame& state, int threadIdx, size_t depth, float parent_board) {
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

    [[maybe_unused]] float reward = 0;

    uct.stats[threadIdx].nodes++;
    uct.stats[threadIdx].rollouts++;

    // NANA_SUFFIX: nothing lies below a leaf yet, so only its eval (or its death) goes up
    [[maybe_unused]] const Return dead{ 0, 0, 0, 1, 1 };

    if (state.game_over) {
#if NANA_SUFFIX
        return dead;
#else
        return -0.0;
#endif
    }

    UCTNode node(state);

    float max_eval = 0.0;

    for (size_t i = 0; i < node.actionCount(); i++) {
        max_eval = std::max(max_eval, node.eval(i));
    }

    // no legal placement: as lost as topping out
    const bool stuck = node.actionCount() == 0;

    if constexpr (NANA_DEPTH_BASELINE) {
        if (!stuck) {
            max_eval = depth_baseline[threadIdx].correct(depth, max_eval, parent_board);
        }
    }

    maybeInsertNode(std::move(node), threadIdx);

    if (stuck) {
#if NANA_SUFFIX
        return dead;
#else
        return 0.0;
#endif
    }

#if NANA_SUFFIX
    return Return{ 0, 0, max_eval, 1, 0 };
#else
    static const NanaConfig& c = config();
    {
        float r = (float)c.app_weight * state.true_app() + (float)c.eval_weight * max_eval;
        reward = std::max(reward, r);
    }

    if (reward_tuning.app_only) {
        return reward;
    }

    // a spike the opponent hasn't tanked yet
    if (state.opponent_jeopardy() > 15) {
        reward += state.opponent_jeopardy() / 20;
    }

    const float x = std::max(reward, 0.0f);
    const float r = x / (x + reward_tuning.scale);
    const float kill = (1.0f - reward_tuning.lambda) * state.expected_kill();
    return kill + (1.0f - kill) * r;
#endif
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