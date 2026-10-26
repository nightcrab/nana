#include "inference.hpp"
#include <atomic>
#include <barrier>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <thread>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif
using namespace opponent_inference;
using Clock = std::chrono::steady_clock;
static double micros(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b-a).count();
}

static void check(const std::string& label, const float* actual, const float* expected, int n) {
    float worst = 0;
    for (int i = 0; i < n; ++i) {
        worst = std::max(worst, std::abs(actual[i]-expected[i]));
        if (!std::isfinite(actual[i]) || std::abs(actual[i]-expected[i]) > 2e-4f+2e-4f*std::abs(expected[i]))
            throw std::runtime_error("Torch parity failed: "+label+" element "+std::to_string(i));
    }
    std::cout << "parity " << label << " max_abs=" << worst << '\n';
}

static void parity(const Model& model, const std::vector<float>& ref) {
    const std::size_t expected = CELLS+CONTEXT_SIZE+HEADS*CELLS+FUSION+HIDDEN+HORIZON*(BINS+HIDDEN);
    if (ref.size() != expected) throw std::runtime_error("reference count mismatch");
    Scratch scratch;
    std::vector<float> features(HEADS*CELLS+CONTEXT_SIZE), conditioning(FUSION), hidden(HIDDEN), gates(3*HIDDEN),
                       recurrent(3*HIDDEN), logits(BINS);
    const float* expected_features = ref.data()+CELLS+CONTEXT_SIZE;
    for (int h = 0; h < HEADS; ++h) model.encode_head(h, ref.data(), features.data()+h*CELLS, scratch);
    check("heads", features.data(), expected_features, HEADS*CELLS);
    std::copy_n(ref.data()+CELLS, CONTEXT_SIZE, features.data()+HEADS*CELLS);
    for (int i = 0; i < FUSION; ++i) conditioning[i] = std::max(0.0f, model.fusion.row(features.data(), i));
    check("fusion", conditioning.data(), expected_features+HEADS*CELLS, FUSION);
    for (int i = 0; i < 4*HIDDEN; ++i) model.prepare_row(i, conditioning.data(), gates.data(), hidden.data());
    const float* expected_hidden = expected_features+HEADS*CELLS+FUSION;
    check("initial", hidden.data(), expected_hidden, HIDDEN);
    const float* step = expected_hidden+HIDDEN;
    int previous = START;
    for (int t = 0; t < HORIZON; ++t) {
        model.step(previous, gates.data(), hidden.data(), logits.data(), recurrent.data());
        check("logits-"+std::to_string(t), logits.data(), step, BINS);
        check("hidden-"+std::to_string(t), hidden.data(), step+BINS, HIDDEN);
        step += BINS+HIDDEN;
        previous = t%8;
    }
}

struct Candidate { std::array<uint8_t, HORIZON> sequence{}; float logp = 0; };
class PhaseBarrier {
    std::barrier<> blocking;
    const int participants;
    const bool spin;
    alignas(64) std::atomic<int> arrived{0};
    alignas(64) std::atomic<unsigned> generation{0};
public:
    PhaseBarrier(int count, bool busy_wait) : blocking(count), participants(count), spin(busy_wait) {}
    void arrive_and_wait() {
        if (!spin) { blocking.arrive_and_wait(); return; }
        const auto phase = generation.load(std::memory_order_acquire);
        if (arrived.fetch_add(1, std::memory_order_acq_rel) == participants-1) {
            arrived.store(0, std::memory_order_relaxed);
            generation.fetch_add(1, std::memory_order_release);
        } else {
            while (generation.load(std::memory_order_acquire) == phase) {
#if defined(__x86_64__) || defined(_M_X64)
                _mm_pause();
#else
                std::this_thread::yield();
#endif
            }
        }
    }
};
struct Workspace {
    Scratch scratch;
    std::vector<float> hidden = std::vector<float>(HIDDEN);
    std::vector<float> recurrent = std::vector<float>(3*HIDDEN);
    std::array<float, BINS> logits{};
};
static uint32_t random_bits(uint32_t& state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state;
}
static void sample(const Model& model, const std::vector<float>& gates, const std::vector<float>& initial,
                   Candidate& candidate, Workspace& workspace, uint32_t state) {
    std::copy(initial.begin(), initial.end(), workspace.hidden.begin());
    int previous = START;
    candidate.logp = 0;
    for (int t = 0; t < HORIZON; ++t) {
        // Always execute all steps, including after death, to keep benchmark work fixed.
        model.step(previous, gates.data(), workspace.hidden.data(), workspace.logits.data(), workspace.recurrent.data());
        const float largest = *std::max_element(workspace.logits.begin(), workspace.logits.end());
        std::array<float, BINS> probabilities;
        float total = 0;
        for (int j = 0; j < BINS; ++j) total += probabilities[j] = std::exp(workspace.logits[j]-largest);
        float u = (random_bits(state) >> 8)*(1.0f/16777216.0f)*total;
        int token = 0;
        for (; token < BINS-1 && u >= probabilities[token]; ++token) u -= probabilities[token];
        if (previous == BINS-1) token = BINS-1;
        else candidate.logp += workspace.logits[token]-largest-std::log(total);
        candidate.sequence[t] = static_cast<uint8_t>(token);
        previous = token;
    }
}

static double quantile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    return values[static_cast<std::size_t>((values.size()-1)*q)];
}

int main(int argc, char** argv) try {
    if (argc < 2) throw std::runtime_error("usage: opponent_infer_bench MODEL_DIR [threads=16] [iterations=200] [samples=32] [warmup=30] [pin=1] [spin=1]");
    const int threads = argc > 2 ? std::stoi(argv[2]) : 16;
    const int iterations = argc > 3 ? std::stoi(argv[3]) : 200;
    const int samples = argc > 4 ? std::stoi(argv[4]) : 32;
    const int warmup = argc > 5 ? std::stoi(argv[5]) : 30;
    const bool pin = argc > 6 ? std::stoi(argv[6]) != 0 : true;
    const bool spin = argc > 7 ? std::stoi(argv[7]) != 0 : true;
    if (threads < 1 || iterations < 1 || samples < 3 || warmup < 0) throw std::runtime_error("invalid arguments");
    const Model model(argv[1]);
    const auto reference = read_floats(std::string(argv[1])+"/reference.bin");
    parity(model, reference);
    std::vector<int> cpus;
#ifdef __linux__
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed)) throw std::runtime_error("cannot read CPU affinity");
    for (int i = 0; i < CPU_SETSIZE; ++i) if (CPU_ISSET(i, &allowed)) cpus.push_back(i);
    if (pin && threads > static_cast<int>(cpus.size())) throw std::runtime_error("more workers than allowed CPUs");
#endif
    std::vector<float> features(HEADS*CELLS+CONTEXT_SIZE), conditioning(FUSION), initial(HIDDEN), gates(3*HIDDEN);
    std::copy_n(reference.data()+CELLS, CONTEXT_SIZE, features.data()+HEADS*CELLS);
    std::vector<Candidate> candidates(samples);
    std::vector<Workspace> workspaces(threads);
    std::vector<int> order(samples), unique;
    unique.reserve(samples);
    std::vector<double> elapsed, conv_times, fusion_times, prepare_times, decode_times, rank_times;
    std::array<std::vector<double>, 4> skews;
    std::array<std::vector<Clock::time_point>, 4> arrivals;
    for (auto& v : arrivals) v.resize(threads);
    PhaseBarrier barrier(threads, spin);
    double checksum = 0;
    std::atomic<bool> affinity_failed{false};
    auto worker = [&](int id) {
#ifdef __linux__
        if (pin) {
            cpu_set_t one;
            CPU_ZERO(&one); CPU_SET(cpus[id], &one);
            if (pthread_setaffinity_np(pthread_self(), sizeof(one), &one)) affinity_failed = true;
        }
#endif
        auto& workspace = workspaces[id];
        for (int iteration = 0; iteration < warmup+iterations; ++iteration) {
            const auto start = Clock::now();
            barrier.arrive_and_wait(); // Include a root-start barrier in end-to-end latency.
            for (int h = id; h < HEADS; h += threads)
                model.encode_head(h, reference.data(), features.data()+h*CELLS, workspace.scratch);
            arrivals[0][id] = Clock::now();
            barrier.arrive_and_wait();
            const auto encoded = Clock::now();
            // Contiguous slices avoid interleaving output cache lines across workers.
            for (int j = FUSION*id/threads; j < FUSION*(id+1)/threads; ++j)
                conditioning[j] = std::max(0.0f, model.fusion.row(features.data(), j));
            arrivals[1][id] = Clock::now();
            barrier.arrive_and_wait();
            const auto fused = Clock::now();
            for (int j = 4*HIDDEN*id/threads; j < 4*HIDDEN*(id+1)/threads; ++j)
                model.prepare_row(j, conditioning.data(), gates.data(), initial.data());
            arrivals[2][id] = Clock::now();
            barrier.arrive_and_wait();
            const auto prepared = Clock::now();
            for (int s = id; s < samples; s += threads)
                sample(model, gates, initial, candidates[s], workspace,
                       1u+static_cast<uint32_t>(iteration)*1664525u+static_cast<uint32_t>(s)*1013904223u);
            arrivals[3][id] = Clock::now();
            barrier.arrive_and_wait();
            const auto decoded = Clock::now();
            if (id == 0) {
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(), [&](int a, int b) { return candidates[a].logp > candidates[b].logp; });
                unique.clear();
                for (int s : order) {
                    bool seen = false;
                    for (int u : unique) if (candidates[u].sequence == candidates[s].sequence) { seen = true; break; }
                    if (!seen) unique.push_back(s);
                    if (unique.size() == 3) break;
                }
                const auto ranked = Clock::now();
                checksum += candidates[order[0]].logp;
                if (iteration >= warmup) {
                    elapsed.push_back(micros(start, ranked));
                    conv_times.push_back(micros(start, encoded));
                    fusion_times.push_back(micros(encoded, fused));
                    prepare_times.push_back(micros(fused, prepared));
                    decode_times.push_back(micros(prepared, decoded));
                    rank_times.push_back(micros(decoded, ranked));
                    for (int stage = 0; stage < 4; ++stage) {
                        auto [lo, hi] = std::minmax_element(arrivals[stage].begin(), arrivals[stage].end());
                        skews[stage].push_back(micros(*lo, *hi));
                    }
                }
            }
            // Workers wait at the next iteration's start barrier while the leader ranks.
        }
    };
    std::vector<std::thread> pool;
    for (int id = 1; id < threads; ++id) pool.emplace_back(worker, id);
    worker(0);
    for (auto& thread : pool) thread.join();
    if (affinity_failed) throw std::runtime_error("CPU affinity setup failed");
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "{\"threads\":" << threads << ",\"samples\":" << samples << ",\"iterations\":" << iterations
              << ",\"pinned\":" << (pin ? "true" : "false") << ",\"spin\":" << (spin ? "true" : "false")
              << ",\"heads\":" << HEADS << ",\"width\":" << WIDTH
              << ",\"hidden\":" << HIDDEN << ",\"horizon\":" << HORIZON
              << ",\"median_us\":" << quantile(elapsed, .5) << ",\"p95_us\":" << quantile(elapsed, .95)
              << ",\"p99_us\":" << quantile(elapsed, .99) << ",\"conv_median_us\":" << quantile(conv_times, .5)
              << ",\"fusion_median_us\":" << quantile(fusion_times, .5)
              << ",\"prepare_median_us\":" << quantile(prepare_times, .5)
              << ",\"decode_median_us\":" << quantile(decode_times, .5)
              << ",\"rank_median_us\":" << quantile(rank_times, .5)
              << ",\"arrival_skew_median_us\":[";
    for (int s = 0; s < 4; ++s) std::cout << (s ? "," : "") << quantile(skews[s], .5);
    std::cout << "],\"checksum\":" << checksum << "}\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n'; return 1;
}
