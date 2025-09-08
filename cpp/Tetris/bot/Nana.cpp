#include "Nana.hpp"

#include "UCT.hpp"
#include "Util/custom_order_max.hpp"

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
#include <fstream>
#include <mutex>

constexpr int LOAD_FACTOR = 6;
#define BENCH


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

		uint32_t getOwner(uint32_t hash) {
			return hash % workers;
		}

		inline bool nodeExists(uint32_t nodeID) {
			// read lock
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
			stats.nodes++;
			nodes_right.insert({ node.id, node });
		};
	};

	static void maybeInsertNode(WorkerParams& params, const UCTNode& node) {
		int owner = node.id % params.workers;
		if (params.threadIdx == owner) {
			params.insertNode(node);
		} else {
			params.mpscs[owner]->enqueue(JobVariant{PutJob{node}}, params.threadIdx);
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

		float max_eval = (*std::ranges::max_element(node.actions, std::ranges::less{}, &Action::eval)).eval;

		maybeInsertNode(params, node);

		if constexpr (search_style == NanaSearchType::NANA) {
			float r = state.true_app() / 3 + max_eval / 2;
			reward = std::max(reward, r);
		}
		if constexpr (search_style == NanaSearchType::CC) {
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
		if(std::holds_alternative<StopJob>(job))
			// do nothing
			return;
		
		if (false && params.mpscs[params.threadIdx]->isempty() && !std::holds_alternative<BackPropJob>(job)) {
			// steal
			params.mpscs[params.threadIdx]->enqueue(std::move(job), params.threadIdx);
		} else {
			// don't steal
			params.mpscs[targetThread]->enqueue(std::move(job), params.threadIdx);
		}
	}

	static void processJob(JobVariant job, WorkerParams &params) {

		if (std::holds_alternative<PutJob>(job)) {
			params.insertNode(std::get<PutJob>(job).node);
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
					.hash_to_ucb_path = std::move(select_job.hash_to_ucb_path),
					.R = reward,
					.depth = state.pieces
				};
				if (select_job.path.empty()) {
					return;
				}

				uint32_t parent_hash = select_job.path.back().hash;

				uint32_t parentIdx = parent_hash % params.workers;

				// send this one back to our parent

				maybeSteal(params, parentIdx, std::move(backprop_job));

				return;
			}

			if (params.nodeExists(hash)) {

				UCTNode& node = params.getNode(hash);
				Action* action = &node.actions[0];

				if constexpr (search_style == NanaSearchType::NANA) {
					if (hash == params.root_state.hash()) {

						action = &node.select(depth);
					} else {

						action = &node.select(depth);
					}
				}
				if constexpr (search_style == NanaSearchType::CC) {

					action = &node.select_SOR(params.rng);
				}

				// Virtual Loss by setting N := N+1
				node.N += 1;
				
				// update ucb history
				node.update_wvt_table(select_job.hash_to_ucb_path);
				
				action->addN();
				
				action->updateTime(params.time);
				
				state.set_move(state.specific_move(action->move));
				
				state.play_moves();
				state.chance_move();

				
				select_job.hash_to_ucb_path.try_emplace(hash, std::vector<WVT>(node.actions.size()));
				node.hash_to_ucb_path.try_emplace(hash, std::vector<WVT>(node.actions.size()));
					
				// append to ucb
				for(auto&action : node.actions) {
					select_job.hash_to_ucb_path.at(hash).at(action.id).N = action.N;
					select_job.hash_to_ucb_path.at(hash).at(action.id).R = action.R;
				}

				uint32_t new_hash = state.hash();

				// Get the owner of the updated state based on the hash
				uint32_t ownerIdx = new_hash % params.workers;

				//Job select_job(state, SELECT, job.path);
				SelectJob new_job{ 
					state, 
					select_job.path, 
					std::move(select_job.hash_to_ucb_path), 
					0.0 
				};

				new_job.path.push_back(HashActionPair(hash, action->id));

				params.stats.deepest_node = std::max(params.stats.deepest_node, (uint64_t)select_job.path.size());

				maybeSteal(params, ownerIdx, std::move(new_job));
			} else {

				float reward = rollout(params, state);

				uint32_t parent_hash = select_job.path.back().hash;

				uint32_t parentIdx = parent_hash % params.workers;

				maybeSteal(params, parentIdx, BackPropJob {
					.state = state,
					.path = select_job.path,
					.hash_to_ucb_path = std::move(select_job.hash_to_ucb_path),
					.R = reward,
					.depth = state.pieces
				});
			}
		} else if (std::holds_alternative<BackPropJob>(job)) {
			BackPropJob& backprop_job = std::get<BackPropJob>(job);
			params.stats.backprop_messages++;
			uint32_t node_hash = backprop_job.path.back().hash;
			UCTNode& node = params.getNode(node_hash);

			float reward = backprop_job.R;

			// Undo Virtual Loss by adding R
			if constexpr (search_style == NanaSearchType::NANA) {
				node.actions[backprop_job.path.back().actionID].addReward(reward);
			}
			if constexpr (search_style == NanaSearchType::CC) {
				if (reward > node.actions[backprop_job.path.back().actionID].R) {
					node.actions[backprop_job.path.back().actionID].R = reward;
				}
			}

			// update ucb history
			node.update_wvt_path(backprop_job.path, reward);

			// get current best action from ucb history

			backprop_job.path.pop_back();

			// root node check
			if (backprop_job.path.empty()) {
				// only one thread acutally does this so its fine
				params.root_state.opponent.reset_rng();
				params.root_state.rng.new_seed();
			}
			// root or were best
			if (backprop_job.path.empty() 
				|| node.ucb_is_current_best(backprop_job.path)
			) {
				auto& node_children = node.hash_to_ucb_path.at(node_hash);
				auto& backprop_children = backprop_job.hash_to_ucb_path.at(node_hash);
				// usb history append
				for(auto&action : node.actions) {
					node_children.at(action.id).N = action.N;
					node_children.at(action.id).R = action.R;

					backprop_children.at(action.id).N = action.N;
					backprop_children.at(action.id).R = action.R;
				}

				//Job select_job(params.root_state, SELECT);

				// give ourself this job
				params.mpscs[params.threadIdx]->enqueue(
					JobVariant{
						SelectJob{ 
							.state = node.state,
							.path = backprop_job.path, 
							.hash_to_ucb_path = std::move(backprop_job.hash_to_ucb_path) }
						}, 
					params.threadIdx
				);

			} else {

				uint32_t parent_hash = backprop_job.path.back().hash;

				uint32_t parentIdx = parent_hash % params.workers;

				// drain stashed rewards

				reward += node.R_buffer;
				node.R_buffer = 0;

				// Job backprop_job(reward, backprop_job.depth, backprop_job.state, BACKPROP, backprop_job.path);
				
				maybeSteal(params, parentIdx, BackPropJob {
					backprop_job.state, 
					backprop_job.path,
					std::move(backprop_job.hash_to_ucb_path),
					reward, 
					backprop_job.depth,
				});
			}
		}
	}

	static void search(std::stop_token stop, WorkerParams params, int thread_idx) {
#ifdef BENCH
		struct guh {
			std::vector<u64> times;
			const WorkerParams &params;
			guh(const WorkerParams &params) : params(params) {}

			~guh() {
				// write times to file with thread index in the name
				std::ofstream file("./bench_" + std::to_string(params.threadIdx) + ".txt");
				for (int i = 0; i < times.size(); i += 2) {
					file << times[i] << " " << times[i + 1] << '\n';
				}
				std::ofstream file2("./backprops_" + std::to_string(params.threadIdx) + ".txt");
				file2 << params.stats.backprop_messages;
			}
			void push_back(u64 time) {
				times.push_back(time);
			}
		}times(params);
#endif
		while (true) {
			
			JobVariant job{StopJob{}};
			// params.mpscs[params.threadIdx]->flush();
			while(params.mpscs[params.threadIdx]->isempty()) {
				if (stop.stop_requested()) {
					params.mpscs[params.threadIdx]->clear();
					return;
				}
			}
			if (stop.stop_requested()) {
				params.mpscs[params.threadIdx]->clear();
				return;
			}


			// check for stop job
			bool need_to_stop = std::ranges::any_of(params.mpscs[params.threadIdx]->flushed_queue, [](auto&& arg){return std::holds_alternative<StopJob>(arg);});
			if(need_to_stop) [[unlikely]] {
				return;
			}
			job = params.mpscs[params.threadIdx]->dequeue();
			
#ifdef BENCH
			struct bench {
				guh& times;

				bench(guh& times) : times(times) {
					times.push_back(std::chrono::steady_clock::now().time_since_epoch().count());
				}
				~bench() {
					auto end = std::chrono::steady_clock::now();
					times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end.time_since_epoch()).count());
				}

			} b(times);
#endif
			if (std::holds_alternative<StopJob>(job)) [[unlikely]] {
				return;
			}

			processJob(std::move(job), params);
		}

	}

};



void Nana::startSearch(const EmulationGame&state, int core_count) {

	start_search_time = std::chrono::high_resolution_clock::now();

	searching = true;

	root_state = state;

	this->core_count = core_count;

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
		queues[rootOwnerIdx]->enqueue(JobVariant{SelectJob{ root_state }}, core_count);
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

	start_search_time = std::chrono::high_resolution_clock::now();

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

	time++;
}


void Nana::printStatistics() {

	auto search_end_time = std::chrono::high_resolution_clock::now();

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
	std::cout << "nodes / second per worker: " << (nodes / (ms / 1000000.0)) / core_count << std::endl;
	std::cout << "backprops / second: " << backprops / (ms / 1000000) << std::endl;
	std::cout << "tree depth: " << depth << std::endl;
}


Move Nana::bestMove() {
    Move best_move;

	if constexpr (search_style == NanaSearchType::NANA) {
		best_move = custom_order_max(
			uct.getNode(root_state.hash()).actions, 
			&Action::N, &Action::R
		).value_or(Action{{},0}).move;
	}

	if constexpr (search_style == NanaSearchType::CC) {
		best_move = custom_order_max(
			uct.getNode(root_state.hash()).actions, 
			&Action::R
		).value_or(Action{{},0}).move;
	}

    return best_move;
}
