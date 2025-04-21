#include "UCT.hpp"
#include "Util/MPSC.hpp"
#include "Util/Distribution.hpp"
#include "Eval.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace Distribution;

static float ln(float x) {
	unsigned int bx = *(unsigned int*)(&x);
	unsigned int ex = bx >> 23;
	signed int t = (signed int)ex - (signed int)127;
	//unsigned int s = (t < 0) ? (-t) : t;
	bx = 1065353216 | (bx & 8388607);
	x = *(float*)(&bx);
	return -1.49278 + (2.11263 + (-0.729104 + 0.10969 * x) * x) * x + 0.6931471806 * t;
}

static float quick_sqrt(const float x)
{
	const float xhalf = 0.5f * x;

	union // get bits for floating value
	{
		float x;
		int i;
	} u;
	u.x = x;
	u.i = 0x5f3759df - (u.i >> 1);  // gives initial guess y0
	return x * u.x * (1.5f - xhalf * u.x * u.x);// Newton step, repeating increases accuracy 
}

void Action::addN() {
	N = N * 0.9999 + 1;
}
void Action::addReward(float reward) {
	R = R * 0.9999 + reward;
}

void Action::updateTime(int newtime) {
	if (time != newtime) {
		N = N / 5;
		R = R / 5;
		time = newtime;
	}
}

// Q, the average return
float Action::Q() {
	return R / N;
}

UCTNode::UCTNode(const EmulationGame &state) {
	std::vector<Piece> raw_actions = state.game.get_possible_piece_placements();

	std::vector<Action> actions;
	actions.reserve(raw_actions.size() * 2);

	std::vector<Stochastic<int>> prior;

	int idx = 0;

	int gheight = state.game.board.get_garbage_height();

	for (auto& raw_action : raw_actions) {
		float eval = Eval::eval_CC(state.game, Move(raw_action, false));
		// Surface abstraction step
		raw_action.position.y -= gheight;
		actions.emplace_back(Move(raw_action, false), idx, eval);
		prior.push_back(Stochastic<int>(idx, eval));
		idx++;
	}

	sort_des(prior);

	for (int rank = 1; rank <= prior.size(); rank++) {
		float prob = 1.0 / quick_sqrt(rank);
		prior[rank - 1].probability = prob;
	}

	prior = normalise(prior);

	for (Stochastic<int>& P : prior) {
		actions[P.value].prior = P.probability;
	}

	this->actions = actions;

	this->id = state.hash();

	this->N = 1;
};

Action& UCTNode::select_r_max() {
	Action* best_action = &actions[0];
	float highest_priority = -1.0;

	for (Action& edge : actions) {
		float priority = std::max(edge.eval, edge.R);
		if (priority > highest_priority) {
			best_action = &edge;
			highest_priority = priority;
		}
	}

	return *best_action;
}



Action& UCTNode::select_uct(int depth) {
	Action* best_action = &actions[0];
	float highest_priority = -2.0;

	/*
		UCT formula, ignores policy.
	*/

	float C = 1.507;

	for (Action& edge : actions) {
		if (edge.N == 0) {
			best_action = &edge;
			break;
		}
		float Q = edge.Q();
		if (edge.N == 0) {
			Q = 0.0;
		}
		float U = 2 * C * quick_sqrt(ln(N) / edge.N);
		float priority = Q + U;

		if (priority > highest_priority) {
			best_action = &edge;
			highest_priority = priority;
		}
	}

	return *best_action;
}


Action& UCTNode::select(int depth) {
	Action* best_action = &actions[0];
	float highest_priority = -2.0;

	/*
		D. Silver's PUCT formula.

		Dynamically selects actions based on both policy prior and pull statistics.

		We modify the formula to explore more at shallow depths and exploit more
		at deeper parts of the tree.
	*/

	constexpr float c_init = 1.25;
	constexpr float c_base = 19652;

	float D = 1 + depth/3;

	float c_puct = ln((N + c_base + 1.0) / c_base) + c_init * D;


	for (Action& edge : actions) {
		float Q = edge.Q();
		if (edge.N == 0) {
			Q = 0.0;
		}
		float U = c_puct * edge.prior * quick_sqrt(N) / (1 + edge.N);
		float priority = Q + U;

		if (priority > highest_priority) {
			best_action = &edge;
			highest_priority = priority;
		}
	}

	return *best_action;
}

Action& UCTNode::select_SOR(RNG &rng) {
	const float k = 0.0;

	std::vector<Stochastic<int>> policy;

	for (int idx = 0; idx < actions.size(); idx++) {
		policy.push_back(Stochastic<int>(idx, std::max(actions[idx].eval, actions[idx].R)));
	}

	sort_des(policy);

	for (int rank = 1; rank <= policy.size(); rank++) {
		float prob = 1.0 / (rank * rank);
		policy[rank - 1].probability = prob + k;
	}

	policy = normalise(policy);

	int id = sample(policy, rng);

	Action &action = actions[id];

	return action;
}

bool UCT::nodeExists(uint32_t nodeID) {
	// node could be on left or right side

	// read lock
	std::shared_lock<std::shared_mutex> lock(mutexes[nodeID % workers]);

	bool exists = nodes_left[nodeID % workers]->find(nodeID) != nodes_left[nodeID % workers]->end();
	exists = exists || (nodes_right[nodeID % workers]->find(nodeID) != nodes_right[nodeID % workers]->end());
	return exists;
}


UCTNode& UCT::getNode(uint32_t nodeID, int threadIdx) {
	// if we're here, the node exists somewhere

	mutexes[nodeID % workers].lock();
	bool on_left_side = !nodes_right[nodeID % workers]->count(nodeID);
	mutexes[nodeID % workers].unlock();

	if (on_left_side) {
		// node is on the left side

		if ((nodeID % workers) != threadIdx) {
			// not the owner, so we're not allowed to write

			mutexes[nodeID % workers].lock();
			UCTNode& node = nodes_left[nodeID % workers]->at(nodeID);
			mutexes[nodeID % workers].unlock();
			return node;
		}
		else {
			// copy from left side to right side
			insertNode(nodes_left[nodeID % workers]->at(nodeID));
		}
	}

	std::shared_lock<std::shared_mutex> lock(mutexes[nodeID % workers]);
	return nodes_right[nodeID % workers]->at(nodeID);
};

// singlethreaded version
UCTNode& UCT::getNode(uint32_t nodeID) {

	if (nodes_right[nodeID % workers]->find(nodeID) == nodes_right[nodeID % workers]->end()) {
		// copy from left side to right side
		insertNode(nodes_left[nodeID % workers]->at(nodeID));
	}

	return nodes_right[nodeID % workers]->at(nodeID);
};


void UCT::insertNode(const UCTNode &node) {
	// insertions always done on right side
	stats[node.id % workers].nodes++;

	// write lock
	std::unique_lock<std::shared_mutex> lock(mutexes[node.id % workers]);

	nodes_right[node.id % workers]->insert({ node.id, node });
};

uint32_t UCT::getOwner(uint32_t hash) {
	return hash % workers;
}

void UCT::collect() {
	// clear out left side
	nodes_left = std::vector<padded_map>(workers);
	// exchange sides
	std::swap(nodes_left, nodes_right);
}

int UCT::map_size() {
	int ret = 0;
	for (auto& map : nodes_left) {
		ret += map->size();
	}
	for (auto& map : nodes_right) {
		ret += map->size();
	}
	return ret;
}