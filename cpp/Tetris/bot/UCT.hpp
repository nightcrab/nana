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

// polynomial forgetting: an edge's returns are weighted i^beta by arrival order (beta = 1: linear),
// on top of the 0.9999 decay, so samples from the subtree's early, flatter policy count less.
// the edge keeps R = Q * M, M its (decayed) count of returns, so R / N still reads as Q with
// virtual visits counted as zero. NANA_FORGET_BETA=0 is the plain decayed mean, through the same code
#ifndef NANA_FORGET
#define NANA_FORGET 0
#endif
#ifndef NANA_FORGET_BETA
#define NANA_FORGET_BETA 1
#endif

// add count returns summing to sum to R (= Q * M), M counting returns
void forgetReturns(float& R, float& M, float sum, int count);

// an edge's search statistics: everything selection reads. kept in their own array per node,
// apart from the rest of the edge (ColdAction), so a select scans 12 bytes per edge
struct EdgeStats {
    float R = 0;
    float N = 0;
    float prior = 1;
#if NANA_FORGET
    float M = 0;
#endif

    // count returns summing to reward, from a backprop
    void addReward(float reward, int count);
    // the same, carried up by a flush (no decay without NANA_FORGET, as before)
    void addCarried(float reward, int count);
    void addN();
    float Q() const;
};

// a move without its minos, which follow from type and rotation (6 bytes instead of 14)
struct PackedMove {
    Coord position;
    RotationDirection rotation;
    PieceType type;
    spinType spin;
    bool null_move;

    static PackedMove pack(const Move& move);
    Move unpack() const;
};

// what an edge needs besides its statistics
struct ColdAction {
    PackedMove move;
    int time = 0;
    float eval = 0;
};

// MP-MCTS (Yang, Aasawat, Yoshizoe 2021) history: for each depth of a job's path,
// the stats of the edge it took and of its strongest siblings when it passed through.
// a backprop can stop and restart selection below a node only if every depth would
// still choose the same edge. full sibling lists don't fit Tetris' branching factor,
// so only the top competitors are kept. (df-UCT kept the best and second best; MP-MCTS all siblings)
#ifndef NANA_HISTORY_COMPETITORS
#define NANA_HISTORY_COMPETITORS 3
#endif
constexpr int HISTORY_COMPETITORS = NANA_HISTORY_COMPETITORS;
// nodes with at least this many visits also store the freshest history for their path
// (MP-MCTS stores it in every node)
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
    // the node at this depth, and the action the path took from it
    uint32_t hash = 0;
    uint32_t chosen = 0;
    // inputs to the node's c_puct
    float node_N = 0;
    int depth = 0;
    // edges[0] is the chosen edge, the rest are its strongest siblings
    uint32_t count = 0;
    HistoryEdge edges[1 + HISTORY_COMPETITORS];
#if NANA_FORGET
    // the chosen edge's M, so returns added to edges[0] are weighted as the edge will weight them
    float chosen_M = 0;
#endif

    // would the node still pick the chosen edge?
    bool stillChosen() const;
};

// state
class UCTNode {
   public:
    UCTNode() {};
    UCTNode(const EmulationGame &state);

    uint32_t id;
    int N;

    // rewards and visits of simulations that stopped backpropagating here (MP-MCTS).
    // the next backprop to leave this node carries them up.
    // with transpositions that may be a different path than the one they came from
    float R_buffer = 0.0;
    int N_buffer = 0;
    // how many returns R_buffer sums
    int M_buffer = 0;

    // the parent and edge this node was last selected from (NANA_FLUSH_AT_END only), so parked
    // buffers can be walked up to the root once a search ends; with transpositions, the latest
    uint32_t parent_hash = 0;
    int parent_action = -1;

    // freshest history for the path that last reached us, only kept on busy nodes
    std::vector<HistoryRow> history;

    // edge i's statistics and the rest of it, in parallel arrays
    std::vector<EdgeStats> edge_stats;
    std::vector<ColdAction> cold;

    size_t actionCount() const { return edge_stats.size(); }
    EdgeStats& stats(size_t i) { return edge_stats[i]; }
    const EdgeStats& stats(size_t i) const { return edge_stats[i]; }
    Move move(size_t i) const { return cold[i].move.unpack(); }
    float eval(size_t i) const { return cold[i].eval; }
    int& time(size_t i) { return cold[i].time; }

    // the first time an edge is used in a new search (tree reuse), its stats are scaled down
    void updateTime(size_t i, int newtime);

    // PUCT, shared by select and the history check so they always agree
    static float c_puct(float node_N, int depth);
    static float priority(float c, float node_N, float R, float N, float prior);

    // the history row for taking edge `chosen` from this node, recorded after its virtual loss
    // (used with select_SOR; select_with_competitors builds it during the select scan)
    HistoryRow historyRow(uint32_t hash, size_t chosen, int depth) const;

    // select (PUCT on the stats before virtual loss) and, in the same scan, the competitors
    // historyRow would record after the virtual loss (node N + 1). fills row.count and
    // row.edges[1..]; the caller fills the rest once the virtual loss is applied
    size_t selectWithCompetitors(int depth, HistoryRow& row) const;

    // these return the index of the chosen edge
    size_t select_uct(int depth);
    size_t select(int depth);
    size_t select_r_max();
    size_t select_SOR(RNG& rng);
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
    // NANA_REPLAY_RESTART: carry a diverged restart's result and the buffers it abandons up a
    // slice of path, adding them to each edge, then park them at the top and stop (no select)
    FLUSH,
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
    // NANA_REPLAY_RESTART: parallel to path, the move taken at each node, so a restart can replay
    // the path from the search root with fresh chance (the nodes, and so their moves, belong to
    // other threads)
    std::vector<PackedMove> moves;

    // BACKPROP: visits on top of the one already counted on the way down,
    // drained from nodes where simulations restarted
    int extra_N = 0;
    // BACKPROP / FLUSH: how many returns R sums
    int M = 0;

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

// queues pass jobs by pointer: a message is 8 bytes instead of a 448-byte Job, and a receiver
// only touches the fields it uses (a backprop never reads the state). search jobs are
// allocated once and reused, so this doesn't allocate per message except for control messages
using Msg = std::unique_ptr<Job>;

class WorkerStatistics {
public:
    uint64_t nodes = 0;
    // rollouts run, terminal ones included: one per simulation (Search::simulation_limit counts these)
    uint64_t rollouts = 0;
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
    uint64_t leaks = 0;           // would-be restarts the leaky gate sent on up (NANA_LEAK)
    uint64_t illegal_moves = 0;       // select steps playing a piece that's neither current nor next to hold
    uint64_t replay_matched = 0;      // NANA_REPLAY_RESTART: replays that reached the restart node
    uint64_t replay_diverged = 0;     // ...that left the path first, so the select went elsewhere
    uint64_t replay_diverge_depth = 0; // depth where they left it, summed
    uint64_t flushes = 0;             // FLUSH messages handled
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

        // fast_seed, so a seeded (fast_seed_reset) search is reproducible
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

    // worker version, threadIdx must own nodeID
    UCTNode& getNode(uint32_t nodeID, int threadIdx);

    UCTNode& getNode(uint32_t nodeID);

    // returns false if the node was already present
    bool insertNode(UCTNode node);

    uint32_t getOwner(uint32_t hash);

    void collect();

    int map_size();
};