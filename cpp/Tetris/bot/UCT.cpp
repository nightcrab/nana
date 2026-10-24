#include "UCT.hpp"
#include "Config.hpp"
#include "Util/MPSC.hpp"
#include "Util/Distribution.hpp"
#include "Eval.hpp"
#include "engine/Utility.hpp"
#include <algorithm>
#include <optional>
#include <cassert>
#include <cstring>
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

// N and R are scaled by this before each update, so N levels off at n_cap
static double decay() {
	static const double d = 1.0 - 1.0 / config().n_cap;
	return d;
}

void EdgeStats::addN() {
	N = N * decay() + 1;
}
void EdgeStats::addReward(float reward) {
	R = R * decay() + reward;
}

void EdgeStats::addCarried(float reward) {
	R += reward;
}

// Q, the average return
float EdgeStats::Q() const {
	return R / N;
}

float rewardScalar(const Return& r) {
	static const NanaConfig& c = config();
	// the attack rate of the returns that lived; those that died are worth 0
	const float rate = r.L > 0 ? r.A / r.L : 0.0f;
	return (float)c.app_weight * rate * (r.n - r.n_dead) + (float)c.eval_weight * r.E;
}

Reward UCTNode::climb(size_t i, Reward reward, bool carried, bool apply) {
#if NANA_SUFFIX
	// the edge's own move: its attack and one piece for each return that lived
	constexpr float gamma = (float)NANA_SUFFIX_DISCOUNT;
	Return v = reward;
	v.A = gamma * v.A + (v.n - v.n_dead) * cold[i].step_attack;
	v.L = gamma * v.L + (v.n - v.n_dead);

	if (apply) {
		Return& s = cold[i].sums;
		if (!carried) {
			const float d = (float)decay();
			s.A *= d;
			s.L *= d;
			s.E *= d;
			s.n *= d;
			s.n_dead *= d;
		}
		s += v;
		edge_stats[i].R = rewardScalar(s);
	}
	return v;
#else
	if (apply) {
		if (carried) {
			edge_stats[i].addCarried(reward);
		}
		else {
			edge_stats[i].addReward(reward);
		}
	}
	return reward;
#endif
}

void UCTNode::updateTime(size_t i, int newtime) {
	if (time(i) != newtime) {
		EdgeStats& edge = stats(i);
		edge.N = edge.N / 5;
		edge.R = edge.R / 5;
#if NANA_SUFFIX
		Return& s = cold[i].sums;
		s.A /= 5;
		s.L /= 5;
		s.E /= 5;
		s.n /= 5;
		s.n_dead /= 5;
#endif
		time(i) = newtime;
	}
}

PackedMove PackedMove::pack(const Move& move) {
	const Piece& piece = move.piece;
	return { piece.position, piece.rotation, piece.type, piece.spin, move.null_move };
}

Move PackedMove::unpack() const {
	return Move(Piece(type, rotation, position, spin), null_move);
}

UCTNode::UCTNode(const EmulationGame &state) {
	std::vector<Piece> raw_actions = state.placements();
	// the smeared movegen occasionally yields a placement with a mino off the board (seen at x = -1),
	// which place_piece would write outside the board's columns
	std::erase_if(raw_actions, [&](const Piece& piece) {
		return Shaktris::Utility::collides(state.game.board, piece);
	});

	std::vector<EdgeStats> edge_stats(raw_actions.size());
	std::vector<ColdAction> cold;
	cold.reserve(raw_actions.size());

	std::vector<Stochastic<int>> prior;
	prior.reserve(raw_actions.size());

	int idx = 0;

	int gheight = state.game.board.get_garbage_height();

	static const bool use_cc_eval = config().eval == "cc";
	std::vector<PieceType> bag_left;
	if (use_cc_eval) {
		bag_left.assign(state.rng.bag.begin() + std::min<size_t>(state.rng.bagiterator, state.rng.bag.size()), state.rng.bag.end());
	}

	// the term over our belief about the HMM opponent, when it has weights and the search tracks one
	static const bool belief_weighted = [] {
		for (double w : config().belief_weights) {
			if (w != 0) return true;
		}
		return false;
	}();
	int meter = 0;
	for (int garbage : state.garbage_meter) {
		meter += garbage;
	}
	std::optional<Eval::BeliefTerm> belief;
	if (use_cc_eval && belief_weighted && state.track_belief) {
		belief.emplace(&state.belief, meter);
	}

	for (auto& raw_action : raw_actions) {
		float eval = use_cc_eval ? Eval::eval_CC_standard(state.game, Move(raw_action, false), bag_left, nullptr, belief ? &*belief : nullptr, meter)
		                         : Eval::eval_CC(state.game, Move(raw_action, false));
		// Surface abstraction step
		raw_action.position.y -= gheight;
		ColdAction c;
		c.move = PackedMove::pack(Move(raw_action, false));
		c.eval = eval;
		// the minos must follow from type and rotation for packing to be lossless
#ifndef NDEBUG
		const Move unpacked = c.move.unpack();
		assert(std::memcmp(&unpacked.piece.minos, &raw_action.minos, sizeof(raw_action.minos)) == 0);
#endif
		cold.push_back(c);
		prior.push_back(Stochastic<int>(idx, eval));
		idx++;
	}

	sort_des(prior);

	static const NanaConfig& c = config();
	for (int rank = 1; rank <= prior.size(); rank++) {
		prior[rank - 1].probability = c.priorWeight(rank);
	}

	prior = normalise(prior);

	for (Stochastic<int>& P : prior) {
		edge_stats[P.value].prior = P.probability;
	}

	this->edge_stats = std::move(edge_stats);
	this->cold = std::move(cold);

	this->id = state.hash();

	this->N = 1;
};

size_t UCTNode::select_r_max() {
	size_t best_action = 0;
	float highest_priority = -1.0;

	for (size_t i = 0; i < actionCount(); i++) {
		float priority = std::max(eval(i), stats(i).R);
		if (priority > highest_priority) {
			best_action = i;
			highest_priority = priority;
		}
	}

	return best_action;
}



size_t UCTNode::select_uct(int depth) {
	size_t best_action = 0;
	float highest_priority = -2.0;

	/*
		UCT formula, ignores policy.
	*/

	float C = 1.507;

	for (size_t i = 0; i < actionCount(); i++) {
		const EdgeStats& edge = stats(i);
		if (edge.N == 0) {
			best_action = i;
			break;
		}
		float Q = edge.Q();
		if (edge.N == 0) {
			Q = 0.0;
		}
		float U = 2 * C * quick_sqrt(ln(N) / edge.N);
		float priority = Q + U;

		if (priority > highest_priority) {
			best_action = i;
			highest_priority = priority;
		}
	}

	return best_action;
}


/*
	D. Silver's PUCT formula.

	Dynamically selects actions based on both policy prior and pull statistics.

	We modify the formula to explore more at shallow depths and exploit more
	at deeper parts of the tree.
*/
float UCTNode::c_puct(float node_N, int depth) {
	static const NanaConfig& c = config();
	const float c_init = (float)c.c_init;
	const float c_base = (float)c.c_base;

	float D = 1 + depth / c.depth_step;

	return ln((node_N + c_base + 1.0) / c_base) + c_init * D;
}

float UCTNode::priority(float c, float node_N, float R, float N, float prior) {
	float Q = N == 0 ? 0.0 : R / N;
	float U = c * prior * quick_sqrt(node_N) / (1 + N);
	return Q + U;
}

size_t UCTNode::select(int depth) {
	size_t best_action = 0;
	float highest_priority = -2.0;

	float c = c_puct(N, depth);

	for (size_t i = 0; i < actionCount(); i++) {
		const EdgeStats& edge = stats(i);
		float p = priority(c, N, edge.R, edge.N, edge.prior);

		if (p > highest_priority) {
			best_action = i;
			highest_priority = p;
		}
	}

	return best_action;
}

// keeps row.edges[1..] the strongest edges seen so far, by p, strongest first.
// best[] holds their priorities; ties keep the earlier edge
static void offerCompetitor(HistoryRow& row, float* best, uint32_t capacity, float p, const EdgeStats& edge) {
	uint32_t i = row.count - 1;
	if (i == capacity && p <= best[i - 1]) {
		return;
	}
	if (i == capacity) {
		i--;
	}
	else {
		row.count++;
	}
	while (i > 0 && best[i - 1] < p) {
		best[i] = best[i - 1];
		row.edges[i + 1] = row.edges[i];
		i--;
	}
	best[i] = p;
	row.edges[i + 1] = { edge.R, edge.N, edge.prior };
}

size_t UCTNode::selectWithCompetitors(int depth, HistoryRow& row) const {
	size_t best_action = 0;
	float highest_priority = -2.0;
	float c = c_puct(N, depth);

	// historyRow runs after the virtual loss, so it ranks with the node's N + 1. keep one
	// more than it records, since the edge we pick (which it skips) may be among them
	const float hist_N = N + 1;
	const float hist_c = c_puct(hist_N, depth);
	constexpr uint32_t capacity = HISTORY_COMPETITORS + 1;
	HistoryEdge ranked[capacity];
	size_t ranked_ids[capacity];
	float best[capacity];
	uint32_t ranked_count = 0;

	for (size_t i = 0; i < actionCount(); i++) {
		const EdgeStats& edge = stats(i);
		float p = priority(c, N, edge.R, edge.N, edge.prior);
		if (p > highest_priority) {
			best_action = i;
			highest_priority = p;
		}

		// same insertion as offerCompetitor, tracking which edge each slot holds
		float h = priority(hist_c, hist_N, edge.R, edge.N, edge.prior);
		uint32_t k = ranked_count;
		if (k == capacity && h <= best[k - 1]) {
			continue;
		}
		if (k == capacity) {
			k--;
		}
		else {
			ranked_count++;
		}
		while (k > 0 && best[k - 1] < h) {
			best[k] = best[k - 1];
			ranked[k] = ranked[k - 1];
			ranked_ids[k] = ranked_ids[k - 1];
			k--;
		}
		best[k] = h;
		ranked[k] = { edge.R, edge.N, edge.prior };
		ranked_ids[k] = i;
	}

	// drop the chosen edge, keep the top HISTORY_COMPETITORS of the rest
	row.count = 1;
	for (uint32_t k = 0; k < ranked_count && row.count < 1 + HISTORY_COMPETITORS; k++) {
		if (ranked_ids[k] == best_action) {
			continue;
		}
		row.edges[row.count++] = ranked[k];
	}
	return best_action;
}

HistoryRow UCTNode::historyRow(uint32_t hash, size_t chosen_id, int depth) const {
	const EdgeStats& chosen = stats(chosen_id);
	HistoryRow row;
	row.hash = hash;
	row.chosen = (uint32_t)chosen_id;
	row.node_N = N;
	row.depth = depth;
	row.edges[0] = { chosen.R, chosen.N, chosen.prior };
	row.count = 1;

	// keep the strongest siblings; the rest are unlikely to overtake before we're back
	float c = c_puct(N, depth);
	float best[HISTORY_COMPETITORS];
	for (size_t i = 0; i < actionCount(); i++) {
		if (i == chosen_id) {
			continue;
		}
		const EdgeStats& edge = stats(i);
		offerCompetitor(row, best, HISTORY_COMPETITORS, priority(c, N, edge.R, edge.N, edge.prior), edge);
	}
	return row;
}

bool HistoryRow::stillChosen() const {
	float c = UCTNode::c_puct(node_N, depth);
	float chosen_p = UCTNode::priority(c, node_N, edges[0].R, edges[0].N, edges[0].prior);
	for (uint32_t i = 1; i < count; i++) {
		if (UCTNode::priority(c, node_N, edges[i].R, edges[i].N, edges[i].prior) > chosen_p) {
			return false;
		}
	}
	return true;
}

size_t UCTNode::select_SOR(RNG &rng) {
	const float k = 0.0;

	std::vector<Stochastic<int>> policy;

	for (size_t idx = 0; idx < actionCount(); idx++) {
		policy.push_back(Stochastic<int>((int)idx, std::max(eval(idx), stats(idx).R)));
	}

	sort_des(policy);

	for (int rank = 1; rank <= policy.size(); rank++) {
		float prob = 1.0 / (rank * rank);
		policy[rank - 1].probability = prob + k;
	}

	policy = normalise(policy);

	return (size_t)sample(policy, rng);
}

bool UCT::nodeExists(uint32_t nodeID) {
	// node could be on left or right side
	bool exists = nodes_left[nodeID % workers]->find(nodeID) != nodes_left[nodeID % workers]->end();
	exists = exists || (nodes_right[nodeID % workers]->find(nodeID) != nodes_right[nodeID % workers]->end());
	return exists;
}


UCTNode& UCT::getNode(uint32_t nodeID, int threadIdx) {
	// shards are owner-only, which is what makes the lock-free access safe
	assert(getOwner(nodeID) == (uint32_t)threadIdx);

	return getNode(nodeID);
};

// owner or singlethreaded version
UCTNode& UCT::getNode(uint32_t nodeID) {
	auto& right = nodes_right[nodeID % workers].obj_;

	auto it = right.find(nodeID);
	if (it != right.end()) {
		return it->second;
	}

	// move from left side to right side
	auto& left = nodes_left[nodeID % workers].obj_;
	auto left_it = left.find(nodeID);
	insertNode(std::move(left_it->second));
	left.erase(left_it);

	return right.at(nodeID);
};


bool UCT::insertNode(UCTNode node) {
	const uint32_t id = node.id;

	// insertions always done on right side
	stats[id % workers].nodes++;

	return nodes_right[id % workers]->try_emplace(id, std::move(node)).second;
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