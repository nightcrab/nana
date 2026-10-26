# Standalone opponent inference

This experiment generates an inference package from `Predictor` without training, checks it
against Torch, and measures native CPU latency without the bot or datagen. It can also export
an existing `conv-heads-gru-v1` board checkpoint with `--checkpoint`.

The exporter emits architecture constants and tensor offsets in `generated.hpp`, FP32 parameters
in `weights.bin`, a tensor/shape manifest, and intermediate Torch outputs in `reference.bin`.
The generated execution plan uses reusable convolution, linear, and GRU kernels. Dimensions are
compile-time constants; changing architecture requires re-exporting and recompiling.

```sh
python scripts/export_opponent_inference.py --out results/opponent-infer-w4
cmake -S cpp/opponent_inference -B out/opponent-infer \
  -DCMAKE_BUILD_TYPE=Release \
  -DMODEL_DIR=/absolute/path/to/results/opponent-infer-w4 \
  -DINFER_MARCH=znver5
cmake --build out/opponent-infer -j2
out/opponent-infer/opponent_infer_bench results/opponent-infer-w4 16 500 32 50 1 1
```

Arguments after model directory are worker count, measured iterations, candidate count, warmup
iterations, CPU pinning (Linux), and spinning barriers. Pass a final `0` for `std::barrier`.
The CMake project is standalone: it does not fetch libraries or build Nana/ShakTris.
For other CPUs use `INFER_MARCH=native`; explicit kernels select AVX-512, AVX2, or portable loops
at compile time. `INFER_SCALAR=ON` disables explicit intrinsics, but compiler autovectorization
is still allowed. AVX2 builds need FMA. There is no runtime ISA dispatch in this prototype.

Every invocation first checks all head features, fusion, initial decoder state, and all GRU
logits/hidden states against Torch with absolute/relative tolerances of 2e-4. This includes
PyTorch's reset-after-hidden-projection GRU equation. The benchmark then uses persistent,
optionally pinned workers and preallocated buffers:

1. Root-start barrier, partition convolutional heads across workers.
2. Head barrier, compute contiguous output slices of dense fusion.
3. Fusion barrier, partition initial-state and constant conditioning gate projections.
4. Preparation barrier, independently sample complete trajectories on each worker.
5. Decoder barrier, leader sorts candidates and retains up to three distinct sequences.

The GRU conditioning projection is computed once per root. Each possible previous-token
embedding's gate projection is precomputed when loading the model. All eight decoding steps
execute even after sampled death; subsequent emissions are absorbing death with zero extra
log loss. This keeps operation counts fixed. Samples have deterministic, independent PRNG
streams; this runtime does not use the trainer's beam search. Deduplication may yield fewer
than three candidates, and sampling does not guarantee finding the highest-probability three.
Scores are absolute sequence log probabilities, never normalized over the retained set.

The timing includes the start barrier, all stages, intermediate synchronization, sampling,
softmax, deduplication, and ranking. It excludes model loading, parity checks, thread creation,
and warmup. The last output line is JSON, with total median/p95/p99 microseconds, median stage
times, and median arrival skew (last minus first worker) at the four stage-end barriers.
Stage medians need not sum to the median total. Ranking time includes sort/dedup/selection.

This is a hot-weight, fixed-board, isolated FP32 benchmark. It does not establish playing
strength, prediction coverage, performance under search cache contention, or INT8 latency.
The kernels are a first SIMD baseline, not the result of randomized autotuning.

`scripts/bench_opponent_inference.sh` runs a fixed sweep on Linux with a 2 GiB virtual-memory
limit, sequential compilation, AVX2/AVX-512 variants, 1/8/16/32 pinned workers, 32 candidates,
500 measured iterations and 50 warmups. It saves compiler flags, CPU topology, process load,
per-run logs and peak RSS. It expects generated `results/opponent-infer-w*` directories and
must run on an otherwise idle machine with at least 32 allowed CPUs. It never starts/stops
EC2, installs dependencies, or changes existing training/datagen directories.

The first target measurements and exact source snapshots are recorded in
`results/opponent-inference-20261026/RUN.md`.
