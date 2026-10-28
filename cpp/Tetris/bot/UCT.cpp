#include "UCT.hpp"
#include "CCTrace.hpp"
#include "Config.hpp"
#include "Util/MPSC.hpp"
#include "Util/Distribution.hpp"
#include "Eval.hpp"
#include "PolicyFeatures.hpp"
#include "engine/Utility.hpp"
#include <algorithm>
#include <cassert>
#include <optional>
#include <cstring>
#include <cmath>
#include <iostream>
#ifdef NANA_SEARCH_FIXTURE
#include "tests/SearchFixture.hpp"
#endif

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

float EdgeStats::Q() const {
	return R / N;
}

Return stepUp(Return v, float step) {
	constexpr float gamma = (float)NANA_SUFFIX_DISCOUNT;
	const float moved = v.n - v.n_dead;
	v.A = gamma * v.A + moved * step;
	v.L = gamma * v.L + moved;
	return v;
}

float rewardScalar(const Return& r) {
	static const NanaConfig& c = config();
	const float rate = r.L > 0 ? r.A / r.L : 0.0f;
	return (float)c.app_weight * rate * (r.n - r.n_dead) + (float)c.eval_weight * r.E;
}

Reward UCTNode::climb(size_t i, Reward reward, bool carried) {
#if !NANA_CC_SEARCH
	Return v = stepUp(reward, cold[i].step_attack);

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
	return v;
#else
	if (carried) {
		edge_stats[i].addCarried(reward);
	}
	else {
		edge_stats[i].addReward(reward);
	}
	return reward;
#endif
}

#if !NANA_CC_SEARCH
void ChildTable::put(uint32_t child, Reward V, float k) {
	if (first.n == 0 || first.hash == child) {
		first = { child, first.n + k, V };
		return;
	}
	if (!more) {
		more = std::make_unique<ChildValue[]>(MORE);
	}
	for (int j = 0; j < MORE; j++) {
		ChildValue& slot = more[j];
		if (slot.n == 0 || slot.hash == child) {
			slot = { child, slot.n + k, V };
			return;
		}
	}
	rest_sum += scaled(V, k);
	rest_n += k;
}

Reward ChildTable::sum(float& n) const {
	n = first.n + rest_n;
	Reward s = scaled(first.V, first.n) + rest_sum;
	if (more) {
		for (int j = 0; j < MORE && more[j].n > 0; j++) {
			n += more[j].n;
			s += scaled(more[j].V, more[j].n);
		}
	}
	return s;
}

float UCTNode::armValue(size_t i, Reward& V) const {
	float n = 0;
	Reward s = cold[i].children.sum(n);
	if (n <= 0) {
		V = Reward{};
		return 0.0f;
	}
	s = stepUp(s, cold[i].step_attack);
	V = scaled(s, 1.0f / n);
	return rewardScalar(V);
}

static bool counts(const ChildTable& t) {
	return NANA_EXPECTIMAX_MIN_N <= 1 ? t.completed > 0 : t.completed >= NANA_EXPECTIMAX_MIN_N;
}

Reward UCTNode::report(size_t i, uint32_t child, Reward V, float k, bool fresh) {
	ChildTable& t = cold[i].children;
#ifdef NANA_SEARCH_FIXTURE
    if (search_fixture::inject_weights && child == search_fixture::hash_base+2) k *= 4;
#endif
	t.put(child, V, k);
	t.completed = fresh ? t.completed * (float)decay() + k : t.completed + k;
	Reward arm;
	const float q = armValue(i, arm);
	edge_stats[i].R = q * t.completed;

	if (counts(t)) {
		if ((int)i == best) {
			if (q >= best_Q) {
				best_Q = q;
				best_V = arm;
			}
			else {
				rescanBest();
			}
		}
		else if (best < 0 || q > best_Q) {
			best = (int)i;
			best_Q = q;
			best_V = arm;
		}
	}
	return value();
}

void UCTNode::rescanBest() {
	best = -1;
	for (size_t j = 0; j < actionCount(); j++) {
		if (!counts(cold[j].children)) {
			continue;
		}
		Reward arm;
		const float q = armValue(j, arm);
		if (best < 0 || q > best_Q) {
			best = (int)j;
			best_Q = q;
			best_V = arm;
		}
	}
}
#endif

void UCTNode::updateTime(size_t i, int newtime) {
	if (time(i) != newtime) {
		EdgeStats& edge = stats(i);
		edge.N = edge.N / 5;
		if constexpr (!cc_search) {
			edge.R = edge.R / 5;
		}
#if !NANA_CC_SEARCH
		Return& s = cold[i].sums;
		s.A /= 5;
		s.L /= 5;
		s.E /= 5;
		s.n /= 5;
		s.n_dead /= 5;
		cold[i].children.completed /= 5;
		if ((int)i == best && !counts(cold[i].children)) {
			rescanBest();
		}
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
    if (UCT::retain_for_dump) dump_state = std::make_unique<EmulationGame>(state);
#ifdef NANA_SEARCH_FIXTURE
    if (state.fixture_active) { search_fixture::initialize(*this, state); return; }
#endif
	std::vector<Piece> raw_actions = state.placements();
	std::erase_if(raw_actions, [&](const Piece& piece) {
		return Shaktris::Utility::collides(state.game.board, piece);
	});

	std::vector<EdgeStats> edge_stats(raw_actions.size());
	std::vector<ColdAction> cold;
	cold.reserve(raw_actions.size());

	// scratch reused across nodes on this thread
	thread_local std::vector<Stochastic<int>> prior;
	thread_local std::vector<char> convex_action;
	prior.clear();
	prior.reserve(raw_actions.size());
	convex_action.clear();
	bool any_convex = false;
	static const NanaConfig& c = config();
	const bool softmax = c.policy == "softmax";
	thread_local std::vector<double> logits;
	logits.clear();
	if (softmax) logits.reserve(raw_actions.size());

	int idx = 0;

	int gheight = state.game.board.get_garbage_height();

	static const EvalMode eval_mode = evalMode();
	const bool use_convex_eval = eval_mode == EvalMode::Convex;
	Eval::Context context{ state.game, eval_mode };
	if (eval_mode != EvalMode::Convex) {
		context.bag_left.assign(state.rng.bag.begin() + std::min<size_t>(state.rng.bagiterator, state.rng.bag.size()), state.rng.bag.end());
	}

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
	if (belief_weighted && state.track_belief) {
		belief.emplace(&state.belief, meter);
		context.belief = &*belief;
	}
	context.meter = meter;

	for (auto& raw_action : raw_actions) {
		double move_reward = 0;
		bool is_convex = true;
		float eval = Eval::evaluate(context, raw_action, &move_reward, &is_convex);
		if (softmax) logits.push_back(PolicyFeatures::logit(state.game, raw_action, c.policy_weights, c.policy_max_well_depth));
		if constexpr (cc_search) {
			edge_stats[idx].R = eval;
		}
		// Surface abstraction step
		raw_action.position.y -= gheight;
		ColdAction c;
		c.move = PackedMove::pack(Move(raw_action, false));
		c.eval = eval;
		c.move_reward = (float)move_reward;
#ifndef NDEBUG
		const Move unpacked = c.move.unpack();
		assert(std::memcmp(&unpacked.piece.minos, &raw_action.minos, sizeof(raw_action.minos)) == 0);
#endif
		cold.push_back(std::move(c));
		if (use_convex_eval) {
			convex_action.push_back(is_convex);
			any_convex |= is_convex;
		}
		prior.push_back(Stochastic<int>(idx, eval));
		idx++;
	}

	if (softmax && !logits.empty()) {
		const double max_logit = *std::max_element(logits.begin(), logits.end());
		double sum = 0;
		for (double& logit : logits) { logit = std::exp(logit - max_logit); sum += logit; }
		for (size_t i = 0; i < logits.size(); ++i) edge_stats[i].prior = float(logits[i] / sum);
	}
	else {
		sort_des(prior);
		// convex actions take the top ranks, and the rest are weighted down unless the node has no convex action
		const bool split = use_convex_eval && any_convex;
		if (split) {
			std::stable_partition(prior.begin(), prior.end(), [&](const Stochastic<int>& p) { return convex_action[p.value] != 0; });
		}
		for (int rank = 1; rank <= prior.size(); rank++) {
			prior[rank - 1].probability = c.priorWeight(rank);
			if (split && !convex_action[prior[rank - 1].value]) {
				prior[rank - 1].probability *= (float)c.nonconvex_prior;
			}
		}

		float prior_sum = 0;
		for (const Stochastic<int>& P : prior) {
			prior_sum += P.probability;
		}
		for (const Stochastic<int>& P : prior) {
			edge_stats[P.value].prior = P.probability / prior_sum;
		}
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

float UCTNode::ccValue() const {
	float best = CC_DEAD;
	for (const EdgeStats& edge : edge_stats) {
		best = std::max(best, edge.R);
	}
	return best;
}

double ccUniform(RNG& rng) {
	return ((double)rng.getRand(1u << 16) * 65536.0 + (double)rng.getRand(1u << 16)) / 4294967296.0;
}

size_t UCTNode::select_CC(RNG& rng, int exclude, float* p_chosen) const {
	if (p_chosen) {
		*p_chosen = 1.0f;
	}
#if NANA_CC_LOCKSTEP
	if (cc_trace::select_hook) {
		return cc_trace::select_hook(*this);
	}
#endif
	static thread_local std::vector<std::pair<float, uint32_t>> ranked;
	static thread_local std::vector<double> weights;
	ranked.clear();
	for (size_t i = 0; i < actionCount(); i++) {
		if (stats(i).R > CC_DEAD) {
			ranked.push_back({ stats(i).R, (uint32_t)i });
		}
	}
	if (ranked.empty()) {
		return 0;
	}
	std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
	const float worst = ranked.back().first;
	weights.clear();
	double total = 0;
	for (size_t rank = 0; rank < ranked.size(); rank++) {
		const double e = std::floor((ranked[rank].first - worst) * 10000.0) + 10;
		const double w = std::floor(e * e / (double)(rank * rank + 1));
		weights.push_back(w);
		total += w;
	}
	// leaving out `exclude` and drawing from the rest keeps the full distribution for a caller that kept
	// `exclude` with its own probability first
	size_t skip = ranked.size();
	for (size_t rank = 0; rank < ranked.size(); rank++) {
		if ((int)ranked[rank].second == exclude) {
			skip = rank;
			break;
		}
	}
	double drawable = total;
	if (skip < ranked.size()) {
		if (total - weights[skip] > 0) {
			drawable = total - weights[skip];
		}
		else {
			skip = ranked.size();
		}
	}
#if NANA_CC_LOCKSTEP
	const double u = cc_trace::uniform();
#else
	const double u = ccUniform(rng);
#endif
	double target = u * drawable;
	size_t pick = ranked.size() - 1;
	if (pick == skip && pick > 0) {
		pick--;
	}
	for (size_t rank = 0; rank < ranked.size(); rank++) {
		if (rank == skip) {
			continue;
		}
		target -= weights[rank];
		if (target < 0) {
			pick = rank;
			break;
		}
	}
	if (p_chosen && total > 0) {
		*p_chosen = (float)(weights[pick] / total);
	}
#if NANA_CC_LOCKSTEP
	cc_trace::Step step{ id, {}, u, (uint32_t)pick };
	for (size_t rank = 0; rank < ranked.size(); rank++) {
		step.ranked.push_back({ ranked[rank].second, ranked[rank].first, weights[rank] });
	}
	cc_trace::steps.push_back(std::move(step));
#endif
	return ranked[pick].second;
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
	assert(getOwner(nodeID) == (uint32_t)threadIdx);

	return getNode(nodeID);
};

UCTNode& UCT::getNode(uint32_t nodeID) {
	auto& right = nodes_right[nodeID % workers].obj_;

	auto it = right.find(nodeID);
	if (it != right.end()) {
		return it->second;
	}

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
