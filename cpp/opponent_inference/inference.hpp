#pragma once
#include "generated.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#if !defined(NANA_INFER_SCALAR) && (defined(__AVX2__) || defined(__AVX512F__))
#include <immintrin.h>
#endif

namespace opponent_inference {
using namespace generated;
constexpr int CELLS = 200, BINS = 9, START = 9;

inline std::vector<float> read_floats(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() < 0 || static_cast<std::size_t>(f.tellg()) % sizeof(float))
        throw std::runtime_error("invalid float file: " + path);
    std::vector<float> values(static_cast<std::size_t>(f.tellg()) / sizeof(float));
    f.seekg(0);
    if (!f.read(reinterpret_cast<char*>(values.data()), values.size()*sizeof(float)))
        throw std::runtime_error("short read: " + path);
    return values;
}

inline float dot(const float* a, const float* b, int n) {
    int i = 0;
    float sum = 0;
#if !defined(NANA_INFER_SCALAR) && defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps();
    for (; i + 16 <= n; i += 16)
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a+i), _mm512_loadu_ps(b+i), acc);
    sum = _mm512_reduce_add_ps(acc);
#elif !defined(NANA_INFER_SCALAR) && defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc);
    float lanes[8];
    _mm256_storeu_ps(lanes, acc);
    for (float x : lanes) sum += x;
#endif
    for (; i < n; ++i) sum += a[i]*b[i];
    return sum;
}

struct Conv { const float *weight, *bias; int in, out, kernel; };
struct Linear {
    const float *weight, *bias;
    int in, out;
    float row(const float* x, int i) const { return dot(weight + i*in, x, in) + bias[i]; }
};
struct Scratch {
    std::vector<float> a, b, residual, padded;
    Scratch() : a(std::max(WIDTH, 3)*CELLS), b(std::max(WIDTH, 3)*CELLS),
                residual(WIDTH*CELLS), padded(std::max(WIDTH, 3)*22*12) {}
};

inline void conv(const Conv& c, const float* x, float* out, Scratch& s,
                 bool relu, const float* residual = nullptr) {
    if (c.kernel == 1) {
        for (int o = 0; o < c.out; ++o) for (int p = 0; p < CELLS; ++p) {
            float v = c.bias[o];
            for (int i = 0; i < c.in; ++i) v += c.weight[o*c.in+i]*x[i*CELLS+p];
            out[o*CELLS+p] = relu ? std::max(v, 0.0f) : v;
        }
        return;
    }
    std::fill(s.padded.begin(), s.padded.begin()+c.in*22*12, 0.0f);
    for (int i = 0; i < c.in; ++i) for (int y = 0; y < 20; ++y)
        std::copy_n(x+i*CELLS+y*10, 10, s.padded.data()+i*22*12+(y+1)*12+1);
    for (int o = 0; o < c.out; ++o) for (int y = 0; y < 20; ++y) {
        float* dst = out+o*CELLS+y*10;
        const float* skip = residual ? residual+o*CELLS+y*10 : nullptr;
#if !defined(NANA_INFER_SCALAR) && defined(__AVX512F__)
        constexpr __mmask16 mask = 0x3ff;
        __m512 acc = _mm512_set1_ps(c.bias[o]);
        for (int i = 0; i < c.in; ++i) for (int ky = 0; ky < 3; ++ky) for (int kx = 0; kx < 3; ++kx) {
            const float* src = s.padded.data()+i*22*12+(y+ky)*12+kx;
            acc = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, src),
                                  _mm512_set1_ps(c.weight[((o*c.in+i)*3+ky)*3+kx]), acc);
        }
        if (skip) acc = _mm512_add_ps(acc, _mm512_maskz_loadu_ps(mask, skip));
        if (relu) acc = _mm512_max_ps(acc, _mm512_setzero_ps());
        _mm512_mask_storeu_ps(dst, mask, acc);
#else
        int begin = 0;
#if !defined(NANA_INFER_SCALAR) && defined(__AVX2__)
        __m256 acc = _mm256_set1_ps(c.bias[o]);
        for (int i = 0; i < c.in; ++i) for (int ky = 0; ky < 3; ++ky) for (int kx = 0; kx < 3; ++kx) {
            const float* src = s.padded.data()+i*22*12+(y+ky)*12+kx;
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(src),
                                  _mm256_set1_ps(c.weight[((o*c.in+i)*3+ky)*3+kx]), acc);
        }
        if (skip) acc = _mm256_add_ps(acc, _mm256_loadu_ps(skip));
        if (relu) acc = _mm256_max_ps(acc, _mm256_setzero_ps());
        _mm256_storeu_ps(dst, acc);
        begin = 8;
#endif
        for (int p = begin; p < 10; ++p) {
            float acc = c.bias[o];
            for (int i = 0; i < c.in; ++i) for (int ky = 0; ky < 3; ++ky) for (int kx = 0; kx < 3; ++kx)
                acc += s.padded[i*22*12+(y+ky)*12+kx+p]*c.weight[((o*c.in+i)*3+ky)*3+kx];
            if (skip) acc += skip[p];
            dst[p] = relu ? std::max(acc, 0.0f) : acc;
        }
#endif
    }
}

struct Model {
    std::vector<float> weights;
    std::array<std::vector<Conv>, HEADS> heads;
    Linear fusion, initial, output;
    const float *wih, *whh, *bih, *bhh, *embedding;
    std::vector<float> token_gates;
    explicit Model(const std::string& directory) : weights(read_floats(directory+"/weights.bin")),
                                                  token_gates(10*3*HIDDEN) {
        std::size_t expected = 0;
        for (auto t : TENSORS) expected = std::max(expected, t.offset+t.count);
        if (weights.size() != expected) throw std::runtime_error("weight count does not match generated architecture");
        for (int h = 0; h < HEADS; ++h) {
            const std::string prefix = "heads."+std::to_string(h)+".";
            heads[h].push_back(convolution(prefix+"0", 3, WIDTH, 3));
            for (int b = 0; b < BLOCKS; ++b) {
                auto base = prefix+std::to_string(b+2)+".layers.";
                heads[h].push_back(convolution(base+"0", WIDTH, WIDTH, 3));
                heads[h].push_back(convolution(base+"2", WIDTH, WIDTH, 3));
            }
            heads[h].push_back(convolution(prefix+std::to_string(BLOCKS+2), WIDTH, 1, 1));
        }
        fusion = linear("fusion.0", HEADS*CELLS+CONTEXT_SIZE, FUSION);
        initial = linear("initial", FUSION, HIDDEN);
        output = linear("output", HIDDEN, BINS);
        wih = tensor("gru.weight_ih_l0"); whh = tensor("gru.weight_hh_l0");
        bih = tensor("gru.bias_ih_l0"); bhh = tensor("gru.bias_hh_l0");
        embedding = tensor("embedding.weight");
        for (int token = 0; token < 10; ++token) for (int j = 0; j < 3*HIDDEN; ++j)
            token_gates[token*3*HIDDEN+j] = bih[j]+dot(wih+j*(EMBEDDING+FUSION),
                                                     embedding+token*EMBEDDING, EMBEDDING);
    }
    const float* tensor(const std::string& name) const {
        for (auto t : TENSORS) if (t.name == name) return weights.data()+t.offset;
        throw std::runtime_error("missing tensor: " + name);
    }
    Conv convolution(const std::string& name, int in, int out, int kernel) const {
        return {tensor(name+".weight"), tensor(name+".bias"), in, out, kernel};
    }
    Linear linear(const std::string& name, int in, int out) const {
        return {tensor(name+".weight"), tensor(name+".bias"), in, out};
    }
    void encode_head(int h, const float* board, float* dest, Scratch& scratch) const {
        std::copy_n(board, CELLS, scratch.a.data());
        for (int y = 0; y < 20; ++y) for (int x = 0; x < 10; ++x) {
            scratch.a[CELLS+y*10+x] = -1.0f+2.0f*x/9;
            scratch.a[2*CELLS+y*10+x] = -1.0f+2.0f*y/19;
        }
        conv(heads[h][0], scratch.a.data(), scratch.b.data(), scratch, true);
        for (int b = 0; b < BLOCKS; ++b) {
            std::copy_n(scratch.b.data(), WIDTH*CELLS, scratch.residual.data());
            conv(heads[h][1+2*b], scratch.b.data(), scratch.a.data(), scratch, true);
            conv(heads[h][2+2*b], scratch.a.data(), scratch.b.data(), scratch, true, scratch.residual.data());
        }
        conv(heads[h].back(), scratch.b.data(), dest, scratch, true);
    }
    void prepare_row(int j, const float* conditioning, float* gates, float* hidden) const {
        if (j >= 3*HIDDEN) hidden[j-3*HIDDEN] = initial.row(conditioning, j-3*HIDDEN);
        else gates[j] = dot(wih+j*(EMBEDDING+FUSION)+EMBEDDING, conditioning, FUSION);
    }
    void step(int previous, const float* conditioning_gates, float* hidden, float* logits,
              float* recurrent) const {
        for (int j = 0; j < 3*HIDDEN; ++j)
            recurrent[j] = dot(whh+j*HIDDEN, hidden, HIDDEN)+bhh[j];
        const float* token = token_gates.data()+previous*3*HIDDEN;
        for (int j = 0; j < HIDDEN; ++j) {
            const float r = 1.0f/(1.0f+std::exp(-(conditioning_gates[j]+token[j]+recurrent[j])));
            const float z = 1.0f/(1.0f+std::exp(-(conditioning_gates[HIDDEN+j]+token[HIDDEN+j]+recurrent[HIDDEN+j])));
            const float n = std::tanh(conditioning_gates[2*HIDDEN+j]+token[2*HIDDEN+j]+r*recurrent[2*HIDDEN+j]);
            hidden[j] = (1-z)*n+z*hidden[j];
        }
        for (int j = 0; j < BINS; ++j) logits[j] = output.row(hidden, j);
    }
};

struct BeamTrajectory {
    std::array<uint8_t, HORIZON> sequence{};
    std::array<float, HIDDEN> hidden{};
    float logp = 0;
};

inline std::vector<BeamTrajectory> trajectories(const Model& model, const float* board, const float* context,
                                                int beam = 128, int topk = 3) {
    if (topk < 1 || beam < topk) throw std::runtime_error("invalid beam size");
    Scratch scratch;
    std::vector<float> features(HEADS*CELLS+CONTEXT_SIZE), conditioning(FUSION), gates(3*HIDDEN);
    for (int h = 0; h < HEADS; ++h) model.encode_head(h, board, features.data()+h*CELLS, scratch);
    std::copy_n(context, CONTEXT_SIZE, features.data()+HEADS*CELLS);
    for (int j = 0; j < FUSION; ++j) conditioning[j] = std::max(0.0f, model.fusion.row(features.data(), j));
    std::vector<BeamTrajectory> current(1);
    for (int j = 0; j < 4*HIDDEN; ++j)
        model.prepare_row(j, conditioning.data(), gates.data(), current[0].hidden.data());
    std::array<float, 3*HIDDEN> recurrent;
    std::array<float, BINS> logits;
    for (int step = 0; step < HORIZON; ++step) {
        std::vector<BeamTrajectory> next;
        next.reserve(current.size()*BINS);
        for (auto candidate : current) {
            const int previous = step ? candidate.sequence[step-1] : START;
            if (previous == BINS-1) {
                candidate.sequence[step] = BINS-1;
                next.push_back(candidate);
                continue;
            }
            model.step(previous, gates.data(), candidate.hidden.data(), logits.data(), recurrent.data());
            const float largest = *std::max_element(logits.begin(), logits.end());
            float sum = 0;
            for (float v : logits) sum += std::exp(v-largest);
            const float normalizer = largest+std::log(sum);
            for (int token = 0; token < BINS; ++token) {
                auto child = candidate;
                child.sequence[step] = (uint8_t)token;
                child.logp += logits[token]-normalizer;
                next.push_back(child);
            }
        }
        const int keep = std::min<int>(beam, next.size());
        std::partial_sort(next.begin(), next.begin()+keep, next.end(),
                          [](const auto& a, const auto& b) { return a.logp > b.logp; });
        next.resize(keep);
        current = std::move(next);
    }
    current.resize(std::min<int>(topk, current.size()));
    return current;
}
}
