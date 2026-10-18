#!/usr/bin/env bash
# Thread-scaling benchmark: TDS-UCT vs MP-MCTS on a many-core Linux machine.
#
# usage: scripts/bench_scaling.sh [max_threads] [reps] [ms_per_run]
#
# Builds nana_bench twice (NANA_MPMCTS=0 and 1, headless, Release) and runs both
# position tests at 1, 2, 4, ... threads up to max_threads, interleaving the two
# builds so clock and thermal drift hit both equally. Also runs nana_msgbench scale,
# which shows how much plain work slows per thread as cores are added.
#
# Writes results/<host>-<date>/{runs.csv,summary.txt,scale.txt}.
# Needs cmake, a C++23 compiler (gcc 13+ or clang 17+) and python3 for the summary.
# Run it with nothing else on the machine; prefer instances without SMT (e.g. AWS c7a),
# or set max_threads to the physical core count.
set -euo pipefail

cd "$(dirname "$0")/.."
max_threads=${1:-$(nproc)}
reps=${2:-3}
ms=${3:-3000}

git submodule update --init --recursive

for mode in 0 1; do
    cmake -S . -B "build-bench-mp$mode" -DCMAKE_BUILD_TYPE=Release -DNANA_GUI=OFF \
        -DCMAKE_CXX_FLAGS="-DNANA_MPMCTS=$mode" > /dev/null
    cmake --build "build-bench-mp$mode" --target nana_bench nana_msgbench -j "$(nproc)" > /dev/null
done
tds=build-bench-mp0/cpp/nana_bench
mp=build-bench-mp1/cpp/nana_bench

out="results/$(hostname)-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out"
{
    echo "host: $(hostname)  cpus: $(nproc)  max_threads: $max_threads  reps: $reps  ms: $ms"
    lscpu 2>/dev/null | grep -E "Model name|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|MHz" || true
    git log -1 --format="commit: %h %s"
} > "$out/machine.txt"

threads=()
for ((n = 1; n < max_threads; n *= 2)); do threads+=("$n"); done
threads+=("$max_threads")

field() { grep "^$1" <<< "$2" | head -1 | awk -F': ' '{print $2}' | tr -d '%'; }

echo "build,test,threads,rep,sims_per_s,nodes_per_s,wait_pct,depth,restarts,root_backprops" > "$out/runs.csv"
for test in 0 1; do
    for n in "${threads[@]}"; do
        for ((r = 1; r <= reps; r++)); do
            for build in tds mp; do
                bin=${!build}
                o=$("$bin" "$n" "$ms" "$test")
                # mean wait over the per-thread profile rows
                wait=$(awk '/^ +[0-9]+ +[0-9.]+%/ { gsub("%", "", $2); s += $2; c++ } END { if (c) printf "%.1f", s / c }' <<< "$o")
                echo "$build,$test,$n,$r,$(field 'simulations / second' "$o"),$(field 'nodes / second' "$o"),$wait,$(field 'tree depth' "$o"),$(field 'restarts' "$o"),$(field 'root backprops' "$o")" >> "$out/runs.csv"
                echo "test $test  threads $n  rep $r  $build done"
            done
        done
    done
done

# plain-work slowdown per thread as cores are added (no search), pinned like the workers
build-bench-mp1/cpp/nana_msgbench scale "$max_threads" | tee "$out/scale.txt"

python3 - "$out/runs.csv" <<'EOF' | tee "$out/summary.txt"
import csv, statistics as st, sys, collections
d = collections.defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    d[(r['build'], r['test'], int(r['threads']))].append(r)
med = lambda rs, k: st.median(float(x[k]) for x in rs if x[k])
for test in sorted({k[1] for k in d}):
    print(f"\ntest {test}: simulations/s, median of reps (speedup vs own 1 thread, mean wait)")
    base = {b: med(d[(b, test, 1)], 'sims_per_s') for b in ('tds', 'mp')}
    for n in sorted({k[2] for k in d if k[1] == test}):
        cells = []
        for b in ('tds', 'mp'):
            rs = d[(b, test, n)]
            s = med(rs, 'sims_per_s')
            cells.append(f"{b} {s / 1e3:8.1f}k ({s / base[b]:5.2f}x) wait {med(rs, 'wait_pct'):5.1f}%")
        ratio = med(d[('mp', test, n)], 'sims_per_s') / med(d[('tds', test, n)], 'sims_per_s')
        print(f"  {n:3d} threads | " + " | ".join(cells) + f" | mp/tds x{ratio:.2f}")
EOF
echo "results in $out"
