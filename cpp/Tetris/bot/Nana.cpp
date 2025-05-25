#include "Nana.hpp"

#include "UCT.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <EmulationGame.hpp>
#include <iostream>
#include <memory>
#include <Move.hpp>
#include <Opponent.hpp>
#include <ostream>
#include <ranges>
#include <stop_token>
#include <thread>
#include <unordered_map>
#include <Util/MPSC.hpp>
#include <util/rng.hpp>
#include <variant>
#include <vector>

namespace Nana_Worker {
	struct WorkerParams {
		std::vector<std::unique_ptr<mpsc<JobVariant>>>& mpscs;
		// UCT stuff
		std::unordered_map<int, UCTNode>& nodes_left;
		std::unordered_map<int, UCTNode>& nodes_right;
		RNG& rng;
		WorkerStatistics& stats;
		EmulationGame& root_state;
		int workers;
		int threadIdx;
		int time;

		inline bool nodeExists(uint32_t nodeID) {
			bool exists = nodes_left.find(nodeID) != nodes_left.end();
			exists = exists || (nodes_right.find(nodeID) != nodes_right.end());
			return exists;
		}

		inline UCTNode& getNode(uint32_t nodeID) {

			if (nodes_right.find(nodeID) == nodes_right.end()) {
				// copy from left side to right side
				insertNode(nodes_left.at(nodeID));
			}

			return nodes_right.at(nodeID);
		}

		inline void insertNode(const UCTNode& node) {
			// insertions always done on right side
			stats.nodes++;

			nodes_right.insert({ node.id, node });
		};
	};

	static void maybeInsertNode(WorkerParams& params, UCTNode node) {
		int owner = node.id % params.workers;
		if (params.threadIdx == owner) {
			params.insertNode(node);
		} else {
			PutJob put_job(node);
			params.mpscs[owner]->enqueue(put_job, params.threadIdx);
		}
	}

	static float rollout(WorkerParams& params, EmulationGame& state) {
		// Rollout using eval.

		float reward = 0;

		params.stats.nodes++;

		if (state.game_over) {
			return -0.0;
		}

		UCTNode node(state);

		maybeInsertNode(params, node);

		float max_eval = 0.0;

		for (auto& action : node.actions) {
			max_eval = std::max(max_eval, action.eval);
		}

		maybeInsertNode(params, node);

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

	static void maybeSteal(WorkerParams& params, int targetThread, JobVariant job) {
		bool is_empty = true;
		for (int i = 0; i < params.mpscs[params.threadIdx]->flushed_queue.size(); i++) {
			JobVariant* job = &params.mpscs[params.threadIdx]->flushed_queue[i];
			if (job != nullptr) {
				if (!std::holds_alternative<StopJob>(*job)) {
					is_empty = false;
					break;
				}
			}
		}
		if (is_empty) {
			// steal
			params.mpscs[params.threadIdx]->enqueue(job, params.threadIdx);
		} else {
			// don't steal
			params.mpscs[targetThread]->enqueue(job, params.threadIdx);
		}
	}

	static void processJob(JobVariant job, WorkerParams &params) {

		if (std::holds_alternative<PutJob>(job)) {
			params.nodes_right.insert({
				std::get<PutJob>(job).node.id,
				std::get<PutJob>(job).node 
			});
			return;
		}

		if (std::holds_alternative<SelectJob>(job)) {
			SelectJob& select_job = std::get<SelectJob>(job);
			params.stats.nodes++;

			EmulationGame& state = select_job.state;

			int depth = state.pieces;

			uint32_t hash = state.hash();

			if (state.game_over) {

				float reward = rollout(params, state);

				//Job backprop_job(reward, state.pieces, state, BACKPROP, select_job.path);
				BackPropJob backprop_job{
					.state = state,
					.path = select_job.path,
					.R = reward,
					.depth = state.pieces
				};
				if (select_job.path.empty()) {
					return;
				}

				uint32_t parent_hash = select_job.path.back().hash;

				uint32_t parentIdx = parent_hash % params.workers;

				// send this one back to our parent

				maybeSteal(params, parentIdx, backprop_job);

				return;
			}

			if (params.nodeExists(hash)) {

				UCTNode& node = params.getNode(hash);
				Action* action = &node.actions[0];

				if constexpr (search_style == NANA) {
					if (hash == params.root_state.hash()) {

						action = &node.select(depth);
					} else {

						action = &node.select(depth);
					}
				}
				if constexpr (search_style == CC) {

					action = &node.select_SOR(params.rng);
				}

				// Virtual Loss by setting N := N+1
				node.N += 1;

				action->addN();

				action->updateTime(params.time);

				state.set_move(state.specific_move(action->move));

				state.play_moves();
				state.chance_move();

				uint32_t new_hash = state.hash();

				// Get the owner of the updated state based on the hash
				uint32_t ownerIdx = new_hash % params.workers;

				//Job select_job(state, SELECT, job.path);
				SelectJob new_job{ state, select_job.path };
				new_job.path.push_back(HashActionPair(hash, action->id));

				params.stats.deepest_node = std::max(params.stats.deepest_node, (uint64_t)select_job.path.size());

				maybeSteal(params, ownerIdx, new_job);
			} else {

				float reward = rollout(params, state);

				uint32_t parent_hash = select_job.path.back().hash;

				uint32_t parentIdx = parent_hash % params.workers;

				//Job backprop_job(reward, state.pieces, state, BACKPROP, select_job.path);
				BackPropJob backprop_job{
					.state = state,
					.path = select_job.path,
					.R = reward,
					.depth = state.pieces
				};
				// send rollout reward to parent, who also owns the arm that got here

				maybeSteal(params, parentIdx, backprop_job);
			}
		} else if (std::holds_alternative<BackPropJob>(job)) {
			BackPropJob& backprop_job = std::get<BackPropJob>(job);
			params.stats.backprop_messages++;
			UCTNode& node = params.getNode(backprop_job.path.back().hash);

			float reward = backprop_job.R;

			// Undo Virtual Loss by adding R
			if constexpr (search_style == NANA) {

				node.actions[backprop_job.path.back().actionID].addReward(reward);

			}
			if constexpr (search_style == CC) {
				if (reward > node.actions[backprop_job.path.back().actionID].R) {
					node.actions[backprop_job.path.back().actionID].R = reward;
				}
			}

			backprop_job.path.pop_back();

			if (backprop_job.path.empty()) {
				// only one thread acutally does this so its fine
				params.root_state.opponent.reset_rng();
				params.root_state.rng.new_seed();

				//Job select_job(params.root_state, SELECT);
				SelectJob sj = SelectJob{ params.root_state };

				// i think this was added for minimum work stuff
				/*
				if (threadIdx == params.getOwner(params.root_state.hash())) {
					// extra scope to prevent notifying the main thread while holding the lock
					// this is to prevent the main thread from waiting on the condition variable while we're holding the lock
					{
						std::scoped_lock lock(min_work_mutex);
						min_work_bool = true;
					}
					min_work_cv.notify_one();
					// notify the main thread that we're done
				}
				*/

				// give ourself this job
				params.mpscs[params.threadIdx]->enqueue(sj, params.threadIdx);

				return;
			}

			bool should_backprop = true;

			if (should_backprop) {

				uint32_t parent_hash = backprop_job.path.back().hash;

				uint32_t parentIdx = parent_hash % params.workers;

				// drain stashed rewards

				reward += node.R_buffer;
				node.R_buffer = 0;

				// Job backprop_job(reward, backprop_job.depth, backprop_job.state, BACKPROP, backprop_job.path);
				BackPropJob bpj{
					backprop_job.state, 
					backprop_job.path,
					reward, 
					backprop_job.depth,
				};
				maybeSteal(params, parentIdx, bpj);
			} else {

				// stash reward and start a new rollout

				node.R_buffer += reward;

				//Job select_job(params.root_state, SELECT);
				SelectJob select_job = {
					.state = params.root_state,
					.path = {},
					.R = 0
				};

				// give ourself this job
				params.mpscs[params.threadIdx]->enqueue(select_job, params.threadIdx);
			}
		}
	}

	static void search(std::stop_token stop, WorkerParams params, int thread_idx) {
#ifdef BENCH
		std::vector<u64> times;
#endif
		while (true) {
			if (stop.stop_requested()) {
				return;
			}

			// Thread waits here until something is in the queue
			JobVariant job = params.mpscs[params.threadIdx]->dequeue();

#if BENCH
			struct bench {
				std::vector<u64>& times;

				bench(std::vector<u64>& times) : times(times) {
					times.push_back(std::chrono::steady_clock::now().time_since_epoch().count());
				}
				~bench() {
					auto end = std::chrono::steady_clock::now();
					times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end.time_since_epoch()).count());
				}

			} b(times);
#endif
			if (std::holds_alternative<StopJob>(job)) {
				return;
			}

			processJob(job, params);
		}

		// write times to file with thread index in the name
#ifdef BENCH
		std::ofstream file("bench_" + std::to_string(threadIdx) + ".txt");
		for (int i = 0; i < times.size(); i += 2) {
			file << times[i] << " " << times[i + 1] << std::endl;
		}
#endif
	}

};


constexpr int LOAD_FACTOR = 6;

void Nana::startSearch(const EmulationGame&state, int core_count) {

	start_search_time = std::chrono::high_resolution_clock::now();

	searching = true;

	root_state = state;

	uct = UCT(core_count);

	uct.insertNode(root_state);

	queues.clear();
	queues.reserve(core_count);

	for (const auto&_ : std::views::iota(0,core_count))
		queues.emplace_back(std::make_unique<mpsc<JobVariant>>(core_count + 1));
	
	worker_threads = std::vector<std::jthread>(core_count);

	int rootOwnerIdx = uct.getOwner(state.hash());

	for (int i = 0; i < LOAD_FACTOR * core_count; i++) {
		root_state.rng.new_seed();
		root_state.opponent.reset_rng();
		queues[rootOwnerIdx]->enqueue(SelectJob{ root_state }, core_count);
	}

	for (const auto& idx : std::views::iota(0, core_count)) {
		worker_threads[idx] = std::jthread(
			Nana_Worker::search,
			worker_stopper.get_token(),
			Nana_Worker::WorkerParams {
				.mpscs = queues,
				.nodes_left = uct.nodes_left[idx].obj_,
				.nodes_right = uct.nodes_right[idx].obj_,
				.rng = uct.rng[idx],
				.stats = uct.stats[idx],
				.root_state = root_state,
				.workers = core_count,
				.threadIdx = idx,
				.time = time
			},
			idx
		);
	}
}


void Nana::continueSearch(const EmulationGame& state) {
	if (state.game_over)
		return;

	start_search_time = std::chrono::steady_clock::now();

	searching = true;

	worker_stopper = std::stop_source();

	root_state = state;

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
	// Initialise worker queues
	for (int i = 0; i < core_count; i++) {
		queues.emplace_back(std::make_unique<mpsc<JobVariant>>(core_count + 1));
	}


	int rootOwnerIdx = uct.getOwner(state.hash());
	for (int j = 0; j < core_count* LOAD_FACTOR; j++) {
		root_state.opponent.reset_rng();
		root_state.rng.new_seed();
		queues[rootOwnerIdx]->enqueue(SelectJob{ root_state }, core_count);
	}

	for (const auto& idx : std::views::iota(0, core_count)) {
		worker_threads[idx] = std::jthread(Nana_Worker::search, 
			worker_stopper.get_token(),
			Nana_Worker::WorkerParams{
				.mpscs = queues,
				.nodes_left = uct.nodes_left[idx].obj_,
				.nodes_right = uct.nodes_right[idx].obj_,
				.rng = uct.rng[idx],
				.stats = uct.stats[idx],
				.root_state = root_state,
				.workers = core_count,
				.threadIdx = idx,
				.time = time
			},
			idx
		);
	}
}


void Nana::endSearch() {

	worker_stopper.request_stop();
	searching = false;

	// stop job
	for (int i = 0; i < core_count; i++) {
		queues[i]->enqueue(StopJob{}, core_count);
	}


	// join threads
	for (auto& thread : worker_threads) {
		thread.join();
	}

	// reset the thread stopper
	worker_stopper = std::stop_source();

	if (uct.map_size() > 200000) {
		uct.collect();
	}

	queues.clear();
}


void Nana::printStatistics() {

	std::chrono::steady_clock::time_point search_end_time = std::chrono::steady_clock::now();

	double ms = std::chrono::duration_cast<std::chrono::microseconds>(search_end_time - start_search_time).count();

	uint64_t nodes = 0;
	uint64_t depth = 0;
	uint64_t backprops = 0;

	for (WorkerStatistics stat : uct.stats) {
		nodes += stat.nodes;
		backprops += stat.backprop_messages;
		depth = std::max(stat.deepest_node, depth);
	}

	std::cout << "nodes: " << nodes << std::endl;
	std::cout << "nodes / second: " << nodes / (ms / 1000000) << std::endl;
	std::cout << "backprops / second: " << backprops / (ms / 1000000) << std::endl;
	std::cout << "tree depth: " << depth << std::endl;
}


Move Nana::bestMove() {

}
