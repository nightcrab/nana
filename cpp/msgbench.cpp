// Microbenchmark for the parts of sending a message between two workers.
// usage: nana_msgbench [messages]
//        nana_msgbench scale [max_threads]   per-thread slowdown of plain work as cores are added

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#endif

#include "EmulationGame.hpp"
#include "Util/Affinity.hpp"
#include "UCT.hpp"
#include "Util/MPSC.hpp"
#include "Util/SPSC.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <memory>
#include <thread>
#include <type_traits>
#include <vector>

using clk = std::chrono::steady_clock;

static std::atomic<uint64_t> g_sink = 0;

static void pin(int cpu) {
	cpu %= (int)std::max(1u, std::thread::hardware_concurrency());
#ifdef _WIN32
	SetThreadAffinityMask(GetCurrentThread(), 1ull << cpu);
#else
	cpu_set_t set;
	CPU_ZERO(&set);
	CPU_SET(cpu, &set);
	pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#endif
}

struct Tiny {
	uint64_t v;
};

struct Blob {
	std::array<uint64_t, sizeof(Job) / 8> v;
};

static EmulationGame proto_state;
static std::vector<HashActionPair> proto_path;

struct TinyKind {
	using T = Tiny;
	static constexpr const char* name = "tiny (8 B)";
	static T make(uint64_t i) { return { i }; }
	static uint64_t consume(const T& m) { return m.v; }
	static void work(T& m) { m.v++; }
};

struct BlobKind {
	using T = Blob;
	static constexpr const char* name = "blob (Job-sized, no heap)";
	static T make(uint64_t i) {
		T m;
		m.v.fill(i);
		return m;
	}
	static uint64_t consume(const T& m) {
		uint64_t s = 0;
		for (size_t k = 0; k < m.v.size(); k += 8) s += m.v[k];
		return s;
	}
	static void work(T& m) {
		for (size_t k = 0; k < m.v.size(); k += 8) m.v[k]++;
	}
};

static void write_job(Job& j) {
	j.state.pieces++;
	j.state.game.board.board[0] ^= 1;
	j.state.markov.meter += 1e-6f;
	if (!j.path.empty()) j.path.back().actionID++;
}

static uint64_t touch_state(const EmulationGame& s) {
	uint64_t x = s.game.board.board[0] + s.pieces;
	if (!s.garbage_meter.empty()) x += s.garbage_meter[0];
	if (!s.buffered_garbage.empty()) x += s.buffered_garbage[0];
	x += (uint64_t)s.markov.meter;
	return x;
}

struct JobKind {
	using T = Job;
	static constexpr const char* name = "Job, state";
	static T make(uint64_t i) {
		Job j(proto_state, SELECT);
		j.extra_N = (int)i;
		return j;
	}
	static uint64_t consume(const T& m) { return touch_state(m.state); }
	static void work(T& m) { write_job(m); }
};

struct JobPathKind {
	using T = Job;
	static constexpr const char* name = "Job, state + path";
	static T make(uint64_t i) {
		Job j(proto_state, SELECT);
		j.path = proto_path;
		j.extra_N = (int)i;
		return j;
	}
	static uint64_t consume(const T& m) {
		uint64_t s = touch_state(m.state);
		for (auto& p : m.path) s += p.hash;
		return s;
	}
	static void work(T& m) { write_job(m); }
};

struct BackpropKind {
	using T = Job;
	static constexpr const char* name = "Job, path only read";
	static T make(uint64_t i) { return JobPathKind::make(i); }
	static uint64_t consume(const T& m) {
		uint64_t s = (uint64_t)rewardScalar(m.R);
		for (auto& p : m.path) s += p.hash;
		return s;
	}
	static void work(T& m) {
		m.R = scaled(m.R, 2.0f);
		if (!m.path.empty()) m.path.back().actionID++;
	}
};

template <class K>
double same_thread(uint64_t n) {
	mpsc<typename K::T> q(1);
	uint64_t sink = 0;
	auto start = clk::now();
	for (uint64_t i = 0; i < n; i++) {
		q.enqueue(K::make(i), 0);
		sink += K::consume(q.dequeue());
	}
	auto end = clk::now();
	g_sink += sink;
	return std::chrono::duration<double, std::nano>(end - start).count() / n;
}

template <class K>
double stream_spsc(uint64_t n, int producer_cpu, int consumer_cpu) {
	rigtorp::SPSCQueue<typename K::T> q(1024);
	std::atomic<bool> go = false;
	clk::time_point end;

	std::thread consumer([&] {
		pin(consumer_cpu);
		while (!go.load()) {}
		uint64_t sink = 0;
		for (uint64_t i = 0; i < n; i++) {
			typename K::T* m;
			while (!(m = q.front())) {}
			sink += K::consume(*m);
			q.pop();
		}
		end = clk::now();
		g_sink += sink;
	});

	pin(producer_cpu);
	auto start = clk::now();
	go = true;
	for (uint64_t i = 0; i < n; i++) {
		q.push(K::make(i));
	}
	consumer.join();
	return std::chrono::duration<double, std::nano>(end - start).count() / n;
}

template <class K>
double stream_mpsc(uint64_t n, int producer_cpu, int consumer_cpu) {
	mpsc<typename K::T> q(1);
	std::atomic<bool> go = false;
	clk::time_point end;

	std::thread consumer([&] {
		pin(consumer_cpu);
		while (!go.load()) {}
		uint64_t sink = 0;
		for (uint64_t i = 0; i < n; i++) {
			sink += K::consume(q.dequeue());
		}
		end = clk::now();
		g_sink += sink;
	});

	pin(producer_cpu);
	auto start = clk::now();
	go = true;
	for (uint64_t i = 0; i < n; i++) {
		q.enqueue(K::make(i), 0);
	}
	consumer.join();
	return std::chrono::duration<double, std::nano>(end - start).count() / n;
}

static double pingpong(uint64_t n, int a_cpu, int b_cpu) {
	rigtorp::SPSCQueue<Tiny> ab(64), ba(64);
	std::atomic<bool> go = false;

	std::thread b([&] {
		pin(b_cpu);
		while (!go.load()) {}
		for (uint64_t i = 0; i < n; i++) {
			Tiny* m;
			while (!(m = ab.front())) {}
			Tiny r = *m;
			ab.pop();
			ba.push(r);
		}
	});

	pin(a_cpu);
	go = true;
	auto start = clk::now();
	for (uint64_t i = 0; i < n; i++) {
		ab.push(Tiny{ i });
		Tiny* m;
		while (!(m = ba.front())) {}
		ba.pop();
	}
	auto end = clk::now();
	b.join();
	return std::chrono::duration<double, std::nano>(end - start).count() / n / 2;
}


template <class K, bool use_mpsc>
double circulate(uint64_t hops, int a_cpu, int b_cpu, bool same_thread, int in_flight = 6) {
	using T = typename K::T;
	using Q = std::conditional_t<use_mpsc, mpsc<T>, rigtorp::SPSCQueue<T>>;
	auto make_q = [] {
		if constexpr (use_mpsc) return std::make_unique<Q>(1);
		else return std::make_unique<Q>(1024);
	};
	auto send = [](Q& q, T&& m) {
		if constexpr (use_mpsc) q.enqueue(std::move(m), 0);
		else q.push(std::move(m));
	};
	auto recv = [](Q& q) -> T {
		if constexpr (use_mpsc) return q.dequeue();
		else {
			T* p;
			while (!(p = q.front())) {}
			T m = std::move(*p);
			q.pop();
			return m;
		}
	};

	auto qa = make_q();
	auto qb = same_thread ? nullptr : make_q();
	Q& first = same_thread ? *qa : *qb;
	for (int i = 0; i < in_flight; i++) send(first, K::make(i));

	std::atomic<bool> go = false;
	auto worker = [&](Q& inbox, Q& outbox, int cpu, uint64_t n) {
		pin(cpu);
		while (!go.load()) {}
		uint64_t sink = 0;
		for (uint64_t i = 0; i < n; i++) {
			T m = recv(inbox);
			sink += K::consume(m);
			K::work(m);
			send(outbox, std::move(m));
		}
		g_sink += sink;
	};

	clk::time_point start, end;
	if (same_thread) {
		go = true;
		start = clk::now();
		worker(*qa, *qa, a_cpu, hops);
		end = clk::now();
	} else {
		std::thread b(worker, std::ref(*qb), std::ref(*qa), b_cpu, hops / 2);
		start = clk::now();
		go = true;
		worker(*qa, *qb, a_cpu, hops / 2);
		b.join();
		end = clk::now();
	}
	return std::chrono::duration<double, std::nano>(end - start).count() * (same_thread ? 1 : 2) / hops;
}

template <class F>
static double median_of(int reps, F f) {
	std::vector<double> v;
	for (int i = 0; i < reps; i++) v.push_back(f());
	std::sort(v.begin(), v.end());
	return v[v.size() / 2];
}

template <class K>
static void row(uint64_t n, int reps) {
	double st = median_of(reps, [&] { return same_thread<K>(n); });
	double spsc_smt = median_of(reps, [&] { return stream_spsc<K>(n, 2, 3); });
	double spsc_core = median_of(reps, [&] { return stream_spsc<K>(n, 2, 4); });
	double mpsc_smt = median_of(reps, [&] { return stream_mpsc<K>(n, 2, 3); });
	double mpsc_core = median_of(reps, [&] { return stream_mpsc<K>(n, 2, 4); });
	std::printf("%-28s %10.1f %10.1f %10.1f %10.1f %10.1f\n", K::name, st, spsc_smt, spsc_core, mpsc_smt, mpsc_core);
}


template <class K>
static void circ_row(uint64_t n, int reps) {
	double st = median_of(reps, [&] { return circulate<K, true>(n, 2, 2, true); });
	double spsc_smt = median_of(reps, [&] { return circulate<K, false>(n, 2, 3, false); });
	double spsc_core = median_of(reps, [&] { return circulate<K, false>(n, 2, 4, false); });
	double mpsc_smt = median_of(reps, [&] { return circulate<K, true>(n, 2, 3, false); });
	double mpsc_core = median_of(reps, [&] { return circulate<K, true>(n, 2, 4, false); });
	std::printf("%-28s %10.1f %10.1f %10.1f %10.1f %10.1f\n", K::name, st, spsc_smt, spsc_core, mpsc_smt, mpsc_core);
}

struct ScaleWorkload {
	const char* name;
	uint64_t (*op)();
};

static uint64_t op_alu() {
	uint64_t x = 0x9E3779B97F4A7C15ull;
	for (int i = 0; i < 4096; i++) {
		x ^= x << 13;
		x ^= x >> 7;
		x ^= x << 17;
	}
	return x;
}

static uint64_t op_alloc() {
	static const size_t sizes[] = { 2048, 2048, 808, 2048, 2048, 2048, 808, 2048, 2048, 3200, 1600, 3200 };
	void* blocks[std::size(sizes)];
	for (size_t i = 0; i < std::size(sizes); i++) {
		blocks[i] = std::malloc(sizes[i]);
		static_cast<volatile char*>(blocks[i])[0] = 1;
	}
	uint64_t s = 0;
	for (void* b : blocks) {
		s += static_cast<volatile char*>(b)[0];
		std::free(b);
	}
	return s;
}

static uint64_t op_movegen() {
	return proto_state.game.get_possible_piece_placements().size();
}

static uint64_t op_node() {
	UCTNode node(proto_state);
	return node.actionCount();
}

static double scale_run(const ScaleWorkload& w, int threads, double seconds) {
	std::atomic<bool> go = false, stop = false;
	std::vector<uint64_t> ops(threads, 0);
	std::vector<std::thread> pool;
	for (int t = 0; t < threads; t++) {
		pool.emplace_back([&, t] {
			pinThreadToCore(t);
			while (!go.load()) {}
			uint64_t count = 0, sink = 0;
			while (!stop.load(std::memory_order_relaxed)) {
				sink += w.op();
				count++;
			}
			ops[t] = count;
			g_sink += sink;
		});
	}
	go = true;
	std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
	stop = true;
	for (auto& th : pool) th.join();

	uint64_t total = 0;
	for (uint64_t c : ops) total += c;
	return seconds * 1e9 * threads / total;
}

static void scale(int max_threads) {
	const ScaleWorkload workloads[] = {
		{ "alu loop", op_alu },
		{ "heap alloc/free (rollout mix)", op_alloc },
		{ "move gen", op_movegen },
		{ "UCTNode (move gen + eval)", op_node },
	};
	std::vector<int> thread_counts;
	for (int n = 1; n < max_threads; n *= 2) thread_counts.push_back(n);
	thread_counts.push_back(max_threads);

	std::printf("ns per op per thread, threads pinned physical cores first, %d cores (slowdown vs 1 thread)\n", physicalCoreCount());
	std::printf("%-32s", "workload");
	for (int n : thread_counts) std::printf(" %16d", n);
	std::printf("\n");
	for (const ScaleWorkload& w : workloads) {
		std::printf("%-32s", w.name);
		double base = 0;
		for (int n : thread_counts) {
			double ns = median_of(3, [&] { return scale_run(w, n, 1.0); });
			if (n == 1) base = ns;
			std::printf(" %9.0f (%.2fx)", ns, ns / base);
		}
		std::printf("\n");
	}
}

int main(int argc, const char** argv) {
	const bool scale_mode = argc > 1 && std::string(argv[1]) == "scale";
	uint64_t n = argc > 1 && !scale_mode ? std::stoull(argv[1]) : 1'000'000;
	const int reps = 5;

	proto_state.game.board.board = { 0b00011111, 0b00011111, 0b00111101, 0b00011011, 0b00111111,
									 0b00011110, 0b00011111, 0b00000111, 0b01111111, 0b11111111 };
	proto_state.game.queue = { PieceType::Z, PieceType::L, PieceType::S, PieceType::S, PieceType::O, PieceType::Empty };
	proto_state.game.hold = PieceType::S;
	proto_state.game.current_piece = PieceType::O;
	for (int i = 0; i < 4; i++) {
		auto moves = proto_state.legal_moves();
		if (moves.empty()) break;
		proto_state.set_move(proto_state.specific_move(moves[0]));
		proto_state.play_moves();
		proto_state.chance_move();
	}
	for (int i = 0; i < 16; i++) proto_path.emplace_back(0x1234567u * (i + 1), i);

	if (scale_mode) {
		int max_threads = argc > 2 ? std::stoi(argv[2]) : std::max(1, physicalCoreCount());
		scale(max_threads);
		return g_sink == 42 ? 1 : 0;
	}

	std::printf("sizeof(Job) = %zu, path entries = %zu, state vectors: buffered %zu, meter %zu\n",
		sizeof(Job), proto_path.size(), proto_state.buffered_garbage.size(), proto_state.garbage_meter.size());
	std::printf("one-way latency, tiny message: smt sibling %.1f ns, other core %.1f ns\n\n",
		median_of(reps, [&] { return pingpong(n / 4, 2, 3); }),
		median_of(reps, [&] { return pingpong(n / 4, 2, 4); }));

	std::printf("ns per message (streaming, median of %d x %llu msgs)\n", reps, (unsigned long long)n);
	std::printf("%-28s %10s %10s %10s %10s %10s\n", "payload", "same thr", "spsc smt", "spsc core", "mpsc smt", "mpsc core");
	row<TinyKind>(n, reps);
	row<BlobKind>(n, reps);
	row<JobKind>(n, reps);
	row<JobPathKind>(n, reps);
	row<BackpropKind>(n, reps);

	std::printf("\nthread-ns per message (6 messages circulating between two threads, moved not rebuilt)\n");
	std::printf("%-28s %10s %10s %10s %10s %10s\n", "payload", "same thr*", "spsc smt", "spsc core", "mpsc smt", "mpsc core");
	circ_row<TinyKind>(n, reps);
	circ_row<BlobKind>(n, reps);
	circ_row<JobPathKind>(n, reps);
	circ_row<BackpropKind>(n, reps);
	std::printf("* same thread uses the mpsc queue\n");

	return g_sink == 42 ? 1 : 0;
}
