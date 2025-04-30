#pragma once
#include <algorithm>
#include <stack>
#include <unordered_map>
#include <vector>
#include <shared_mutex>

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
    float R_buffer = 0.0;

    std::vector<Action> actions;

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
};

class Job {
   public:
    // STOP job
    Job()
        : R(0.0), state(EmulationGame()), type(STOP){};

    // SELECT job
    Job(const EmulationGame& state, JobType type)
        : R(0.0), state(state), type(type){};

    // SELECT job
    Job(const EmulationGame& state, JobType type, const std::vector<HashActionPair> &path)
        : R(0.0), state(state), type(type), path(path){};

    // BACKPROP job
    Job(float R, int depth, const EmulationGame& state, JobType type, const std::vector<HashActionPair> &path)
        : R(R), depth(depth), state(state), type(type), path(path){};

    // PUT job
    Job(UCTNode& node, JobType type)
        : R(0.0), type(type), node(node){};

    float R;
    int depth;
    EmulationGame state;
    JobType type;
    UCTNode node;

    // for going backwards

    std::vector<HashActionPair> path;
};

class WorkerStatistics {
public:
    uint64_t nodes = 0;
    uint64_t deepest_node = 0;
    uint64_t backprop_messages = 0;
};

// one of these shared by all threads
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
        this->mutexes = std::vector<std::shared_mutex>(workers);
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
    std::vector<std::shared_mutex> mutexes;

    std::vector<RNG> rng;
    std::vector<WorkerStatistics> stats;

    bool nodeExists(uint32_t nodeID);

    UCTNode& getNode(uint32_t nodeID, int threadIdx);

    UCTNode& getNode(uint32_t nodeID);

    void insertNode(const UCTNode &node);

    uint32_t getOwner(uint32_t hash);

    void collect();

    int map_size();
};