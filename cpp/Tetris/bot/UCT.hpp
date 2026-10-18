#pragma once
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

class Action {
   public:
    Action(Move move, uint32_t id) {
        this->move = move;
        this->N = 0;
        this->R = 0;
        this->prior = 1;
        this->eval = 0.0;
        this->id = id;
    }
    Action(Move move, uint32_t id, float eval) {
        this->move = move;
        this->N = 0;
        this->R = 0;
        this->prior = 1;
        this->eval = eval;
        this->id = id;
    }

    void addReward(float reward);
    void addN();
    float Q();
    void updateTime(int newtime);

    static const int WINDOW_SIZE = 1000;
    Move move;
    float R;
    float prior;
    float eval;
    float N;
    int time;
    uint32_t id;
};

// MP-MCTS (Yang, Aasawat, Yoshizoe 2021) history: for each depth of a job's path,
// the stats of the edge it took and of its strongest siblings when it passed through.
// a backprop can stop and restart selection below a node only if every depth would
// still choose the same edge. full sibling lists don't fit Tetris' branching factor,
// so only the top competitors are kept.
constexpr int HISTORY_COMPETITORS = 3;
// nodes with at least this many visits also store the freshest history for their path
constexpr int HISTORY_MIN_VISITS = 32;

struct HistoryEdge {
    float R = 0;
    float N = 0;
    float prior = 0;
};

struct HistoryRow {
    // the node at this depth, and the action the path took from it
    uint32_t hash = 0;
    uint32_t chosen = 0;
    // inputs to the node's c_puct
    float node_N = 0;
    int depth = 0;
    // edges[0] is the chosen edge, the rest are its strongest siblings
    uint32_t count = 0;
    HistoryEdge edges[1 + HISTORY_COMPETITORS];

    // would the node still pick the chosen edge?
    bool stillChosen() const;
};

// state
class UCTNode {
   public:
    UCTNode() {};
    UCTNode(const EmulationGame &state);
    UCTNode(const std::vector<Action>& p_actions, int id, int N) {
        this->actions = p_actions;
        for (int i = 0; i < actions.size(); i++) {
            actions[i].id = i;
        }
        this->id = id;
        this->N = N;
    }

    uint32_t id;
    int N;

    // rewards and visits of simulations that stopped backpropagating here (MP-MCTS).
    // the next backprop to leave this node carries them up.
    // with transpositions that may be a different path than the one they came from
    float R_buffer = 0.0;
    int N_buffer = 0;

    // freshest history for the path that last reached us, only kept on busy nodes
    std::vector<HistoryRow> history;

    std::vector<Action> actions;

    // PUCT, shared by select and the history check so they always agree
    static float c_puct(float node_N, int depth);
    static float priority(float c, float node_N, float R, float N, float prior);

    // the history row for taking `chosen` from this node, recorded after its virtual loss
    HistoryRow historyRow(uint32_t hash, const Action& chosen, int depth) const;

    Action& select_uct(int depth);
    Action& select(int depth);
    Action& select_r_max();
    Action& select_SOR(RNG& rng);
};

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
    // a rollout task: expand and evaluate the state wherever it runs,
    // then send the node to its owner and the reward to the parent's owner.
    // made by an owner deferring a new leaf, and handed to other threads on request
    EXPAND,
    // work requesting: ask a thread for a spare rollout; it answers with an
    // EXPAND, or forwards the request, or returns it as NO_WORK once everyone was asked
    REQUEST,
    NO_WORK,
    JOB_TYPE_COUNT,
};

// a ready-made state to copy into jobs that don't use theirs, since
// default-constructing an EmulationGame seeds two RNGs from std::random_device
inline const EmulationGame& blank_state() {
    static const EmulationGame blank{};
    return blank;
}

class Job {
   public:
    // STOP job
    Job()
        : R(0.0), state(EmulationGame()), type(STOP){};

    // SELECT job
    Job(EmulationGame state, JobType type)
        : R(0.0), state(std::move(state)), type(type){};

    // PUT job
    Job(UCTNode node, JobType type)
        : R(0.0), state(blank_state()), type(type), node(std::make_unique<UCTNode>(std::move(node))){};

    // control job (REQUEST, NO_WORK)
    explicit Job(JobType type)
        : R(0.0), state(blank_state()), type(type){};

    // move-only: jobs are handed between threads, never duplicated
    Job(Job&&) noexcept = default;
    Job& operator=(Job&&) noexcept = default;
    Job(const Job&) = delete;
    Job& operator=(const Job&) = delete;

    float R;
    int depth;
    EmulationGame state;
    JobType type;
    // only set for PUT jobs
    std::unique_ptr<UCTNode> node;

    // for going backwards

    std::vector<HashActionPair> path;

    // parallel to path: the MP-MCTS history row for each depth, and the state at each node,
    // so a backprop that stops partway can restart selection from there
    std::vector<HistoryRow> history;
    std::vector<EmulationGame> snapshots;

    // BACKPROP: visits on top of the one already counted on the way down,
    // drained from nodes where simulations restarted
    int extra_N = 0;

    // stamped on every send, so receivers keep a picture of everyone's load (work units)
    uint16_t sender = UINT16_MAX;
    int sender_load = 0;

    // REQUEST / NO_WORK: who asked, how loaded they were, and who has been asked
    int requester = -1;
    int requester_load = 0;
    uint64_t visited = 0;
    // EXPAND sent in answer to a REQUEST
    bool is_reply = false;

    // NANA_BENCH: TSC when last sent, to measure how long messages sit in queues (0 = not timed)
    uint64_t sent_at = 0;
};

class WorkerStatistics {
public:
    uint64_t nodes = 0;
    uint64_t deepest_node = 0;
    uint64_t backprop_messages = 0;
    // expansions of a node we already had (two rollouts raced for the same new leaf)
    uint64_t duplicate_puts = 0;

    // work requesting
    uint64_t requests = 0;        // REQUESTs this thread sent
    uint64_t forwards = 0;        // REQUESTs this thread passed on
    uint64_t gave = 0;            // REQUESTs this thread answered with a rollout
    uint64_t gave_hops = 0;       // threads asked before an answer, summed over gave
    uint64_t gave_cross_l3 = 0;   // ...of which went to a thread on another L3 cache (chiplet)
    uint64_t nones = 0;           // REQUESTs that came back unanswered
    uint64_t received = 0;        // rollouts this thread got from a REQUEST
    uint64_t late = 0;            // ...of which arrived after we'd already run dry
    uint64_t deferred = 0;        // new leaves queued as rollouts instead of run inline

    // MP-MCTS
    uint64_t restarts = 0;        // backprops that stopped and restarted selection below the root
    uint64_t restart_depth = 0;   // depth of those restarts, summed
    uint64_t root_backprops = 0;  // backprops that reached the root
    uint64_t cycles = 0;          // selects that reached a node already on their own path
    uint64_t depth_capped = 0;    // selects treated as leaves because the path was too long

    // profile (NANA_BENCH only), in TSC cycles, indexed by JobType
    uint64_t wait_cycles = 0;
    uint64_t job_cycles[JOB_TYPE_COUNT] = {};
    uint64_t job_count[JOB_TYPE_COUNT] = {};
    uint64_t rollout_cycles = 0;
    uint64_t rollout_count = 0;
    // time messages spent queued before this thread picked them up
    uint64_t queue_cycles = 0;
    uint64_t queue_count = 0;
    // selects at the root and depth 1, where every simulation (TDS) or many (MP) pass through
    uint64_t top_cycles = 0;
    uint64_t top_count = 0;
};

// one of these shared by all threads
// each shard (nodes_left/right[i]) is only ever touched by worker i,
// so no locking is needed; other threads talk to it through messages
class UCT {
   public:

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

        auto devrng = std::random_device();
        for (auto& thread_rng : this->rng) {
            thread_rng.PPTRNG = devrng();
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

    // worker version, threadIdx must own nodeID
    UCTNode& getNode(uint32_t nodeID, int threadIdx);

    UCTNode& getNode(uint32_t nodeID);

    // returns false if the node was already present
    bool insertNode(UCTNode node);

    uint32_t getOwner(uint32_t hash);

    void collect();

    int map_size();
};