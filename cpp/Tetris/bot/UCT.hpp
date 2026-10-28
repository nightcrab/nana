#pragma once
#include "Util/FastSeed.hpp"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <stack>
#include <unordered_map>
#include <vector>

#include "EmulationGame.hpp"
#include "Util/MPSC.hpp"
#include "Move.hpp"
#include "util/rng.hpp"

/*
        Data structures for performing multithreaded UCT search.
*/

class UCT;

struct EdgeStats {
    float R = 0;
    float N = 0;
    float prior = 1;

    void addReward(float reward);
    void addCarried(float reward);
    void addN();
    float Q() const;
};

#ifndef NANA_CC_SEARCH
#define NANA_CC_SEARCH 0
#endif
constexpr bool cc_search = NANA_CC_SEARCH;
constexpr float CC_DEAD = -1e9f;

// The CC search's version of the MP-MCTS restart: a backup that settles below the root restarts through
// a replay that redraws chance and each sampled action, instead of selecting again at the root's owner.
#ifndef NANA_CC_RESTART
#define NANA_CC_RESTART NANA_CC_SEARCH
#endif
constexpr bool cc_restart = NANA_CC_RESTART;

#ifndef NANA_SUFFIX_DISCOUNT
#define NANA_SUFFIX_DISCOUNT 1.0
#endif


struct Return {
    float A = 0;
    float L = 0;
    float E = 0;
    float n = 0;
    float n_dead = 0;

    Return& operator+=(const Return& o) {
        A += o.A; L += o.L; E += o.E; n += o.n; n_dead += o.n_dead;
        return *this;
    }
    Return& operator-=(const Return& o) {
        A -= o.A; L -= o.L; E -= o.E; n -= o.n; n_dead -= o.n_dead;
        return *this;
    }
    friend Return operator+(Return a, const Return& b) { return a += b; }
};

// The regular search backs up suffix returns by expectimax; the CC search backs up Cold Clear's scalar values.
#if NANA_CC_SEARCH
using Reward = float;
#else
using Reward = Return;
#endif

#ifndef NANA_EXPECTIMAX_MIN_N
#define NANA_EXPECTIMAX_MIN_N 1
#endif

inline float scaled(float r, float k) { return r * k; }
inline Return scaled(Return r, float k) {
    r.A *= k; r.L *= k; r.E *= k; r.n *= k; r.n_dead *= k;
    return r;
}

struct ChildValue {
    uint32_t hash = 0;
    float n = 0;
    Reward V{};
};

struct ChildTable {
    static constexpr int MORE = 7;
    ChildValue first;
    std::unique_ptr<ChildValue[]> more;
    float rest_n = 0;
    Reward rest_sum{};
    float completed = 0;

    void put(uint32_t child, Reward V, float k);
    Reward sum(float& n) const;
};

inline float rewardScalar(float r) { return r; }
float rewardScalar(const Return& r);
Return stepUp(Return v, float step);
inline bool rewardEmpty(float r) { return r == 0; }
inline bool rewardEmpty(const Return& r) { return r.n == 0; }

struct PackedMove {
    Coord position;
    RotationDirection rotation;
    PieceType type;
    spinType spin;
    bool null_move;

    static PackedMove pack(const Move& move);
    Move unpack() const;
};

struct ColdAction {
    PackedMove move;
    int time = 0;
    float eval = 0;
    float move_reward = 0;
#if !NANA_CC_SEARCH
    float step_attack = 0;
    Return sums;
    ChildTable children;
#endif
};

#ifndef NANA_HISTORY_COMPETITORS
#define NANA_HISTORY_COMPETITORS 3
#endif
constexpr int HISTORY_COMPETITORS = NANA_HISTORY_COMPETITORS;
#ifndef NANA_HISTORY_MIN_VISITS
#define NANA_HISTORY_MIN_VISITS 32
#endif
constexpr int HISTORY_MIN_VISITS = NANA_HISTORY_MIN_VISITS;

struct HistoryEdge {
    float R = 0;
    float N = 0;
    float prior = 0;
};

struct HistoryRow {
    uint32_t hash = 0;
    uint32_t chosen = 0;
    float node_N = 0;
    int depth = 0;
    uint32_t count = 0;
    HistoryEdge edges[1 + HISTORY_COMPETITORS];

    bool stillChosen() const;
};

// state
class UCTNode {
   public:
    UCTNode() {};
    UCTNode(const EmulationGame &state);

    uint32_t id;
    int N;

    // Only retained for emulator tree dumps; the first state represented by this node.
    std::unique_ptr<EmulationGame> dump_state;

    Reward R_buffer{};
    int N_buffer = 0;

    uint32_t parent_hash = 0;
    int parent_action = -1;

    std::vector<HistoryRow> history;

    std::vector<EdgeStats> edge_stats;
    std::vector<ColdAction> cold;

    size_t actionCount() const { return edge_stats.size(); }
    EdgeStats& stats(size_t i) { return edge_stats[i]; }
    const EdgeStats& stats(size_t i) const { return edge_stats[i]; }
    Move move(size_t i) const { return cold[i].move.unpack(); }
    float eval(size_t i) const { return cold[i].eval; }
    int& time(size_t i) { return cold[i].time; }

    void updateTime(size_t i, int newtime);

    Reward climb(size_t i, Reward reward, bool carried);

#if !NANA_CC_SEARCH
    Reward leaf_V{};
    int best = -1;
    float best_Q = 0;
    Reward best_V{};

    Reward value() const { return best >= 0 ? best_V : leaf_V; }
    Reward report(size_t i, uint32_t child, Reward V, float k, bool fresh);
    void rescanBest();
    float armValue(size_t i, Reward& V) const;
#endif

    static float c_puct(float node_N, int depth);
    static float priority(float c, float node_N, float R, float N, float prior);

    HistoryRow historyRow(uint32_t hash, size_t chosen, int depth) const;

    size_t selectWithCompetitors(int depth, HistoryRow& row) const;


    size_t select_uct(int depth);
    size_t select(int depth);
    size_t select_r_max();
    size_t select_SOR(RNG& rng);
    float ccValue() const;
    // Samples an action by Cold Clear's rank weights, never `exclude` unless nothing else is alive.
    // `p_chosen` gets the pick's probability under the full distribution, `exclude` included.
    size_t select_CC(RNG& rng, int exclude = -1, float* p_chosen = nullptr) const;
};

double ccUniform(RNG& rng);

class HashActionPair {
   public:
    HashActionPair(uint32_t hash, int actionID) {
        this->hash = hash;
        this->actionID = actionID;
    }
    uint32_t hash;
    int actionID;
};

enum JobType {
    SELECT,
    BACKPROP,
    STOP,
    PUT,
    EXPAND,
    REQUEST,
    NO_WORK,
    FLUSH,
    JOB_TYPE_COUNT,
};

inline const EmulationGame& blank_state() {
    static const EmulationGame blank{};
    return blank;
}

class Job {
   public:
    // STOP job
    Job()
        : R(), state(EmulationGame()), type(STOP){};

    // SELECT job
    Job(EmulationGame state, JobType type)
        : R(), state(std::move(state)), type(type){};

    Job(UCTNode node, JobType type)
        : R(), state(blank_state()), type(type), node(std::make_unique<UCTNode>(std::move(node))){};

    explicit Job(JobType type)
        : R(), state(blank_state()), type(type){};

    Job(Job&&) noexcept = default;
    Job& operator=(Job&&) noexcept = default;
    Job(const Job&) = delete;
    Job& operator=(const Job&) = delete;

    Reward R;
    int depth;
    EmulationGame state;
    JobType type;
    std::unique_ptr<UCTNode> node;

    // for going backwards

    std::vector<HashActionPair> path;

    std::vector<HistoryRow> history;
    std::vector<EmulationGame> snapshots;
    std::vector<PackedMove> moves;
    std::vector<float> steps;
    // CC search: each path step's sampling probability, and an action the next selection must skip
    std::vector<float> probs;
    int exclude = -1;

    int extra_N = 0;

    uint32_t child_hash = 0;

    float parent_board = 0;

    uint16_t sender = UINT16_MAX;
    int sender_load = 0;

    int requester = -1;
    int requester_load = 0;
    uint64_t visited = 0;
    bool is_reply = false;

    uint64_t sent_at = 0;
};

using Msg = std::unique_ptr<Job>;

class WorkerStatistics {
public:
    uint64_t nodes = 0;
    uint64_t rollouts = 0;
    uint64_t deepest_node = 0;
    uint64_t backprop_messages = 0;
    uint64_t duplicate_puts = 0;

    uint64_t requests = 0;
    uint64_t forwards = 0;
    uint64_t gave = 0;
    uint64_t gave_hops = 0;
    uint64_t gave_cross_l3 = 0;
    uint64_t nones = 0;
    uint64_t received = 0;
    uint64_t late = 0;
    uint64_t deferred = 0;

    uint64_t restarts = 0;
    uint64_t restart_depth = 0;
    uint64_t root_backprops = 0;
    uint64_t illegal_moves = 0;
    uint64_t replay_matched = 0;
    uint64_t replay_diverged = 0;
    uint64_t replay_diverge_depth = 0;
    uint64_t flushes = 0;
    uint64_t cycles = 0;
    uint64_t depth_capped = 0;

    uint64_t wait_cycles = 0;
    uint64_t job_cycles[JOB_TYPE_COUNT] = {};
    uint64_t job_count[JOB_TYPE_COUNT] = {};
    uint64_t rollout_cycles = 0;
    uint64_t rollout_count = 0;
    uint64_t queue_cycles = 0;
    uint64_t queue_count = 0;
    uint64_t top_cycles = 0;
    uint64_t top_count = 0;
};

// one of these shared by all threads
class UCT {
   public:

    // Set before starting workers. Keeps states and disables between-move collection.
    static inline bool retain_for_dump = false;

    struct padded_map {
        using T = std::unordered_map<int, UCTNode>;
        char padding[64];
        T obj_;

        T* operator->() { return &obj_; }
        const T* operator->() const { return &obj_; }
    };

    UCT() = default;
    UCT(int workers) {
        this->workers = workers;
        this->nodes_left = std::vector<padded_map>(workers);
        this->nodes_right = std::vector<padded_map>(workers);
        this->rng = std::vector<RNG>(workers);
        this->stats = std::vector<WorkerStatistics>(workers);

        for (auto& thread_rng : this->rng) {
            thread_rng.PPTRNG = fast_seed();
        }
    }
    ~UCT() = default;

    // Delete copy constructor and copy assignment operator
    UCT(const UCT&) = delete;
    UCT& operator=(const UCT&) = delete;

    // Allow move construction and move assignment
    UCT(UCT&&) noexcept = default;
    UCT& operator=(UCT&&) noexcept = default;


    int workers = 0;
    int size = 0;

    std::vector<padded_map> nodes_left;
    std::vector<padded_map> nodes_right;

    std::vector<RNG> rng;
    std::vector<WorkerStatistics> stats;

    bool nodeExists(uint32_t nodeID);

    UCTNode& getNode(uint32_t nodeID, int threadIdx);

    UCTNode& getNode(uint32_t nodeID);

    bool insertNode(UCTNode node);

    uint32_t getOwner(uint32_t hash);

    void collect();

    int map_size();
};
