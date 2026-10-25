#!/usr/bin/env bash
# Thread-scaling benchmark for MP-MCTS (optionally against TDS-UCT) on a many-core Linux machine.
# usage: scripts/bench_scaling.sh [max_threads] [reps] [ms_per_run]
set -euo pipefail

cd "$(dirname "$0")/.."
max_threads=${1:-$(nproc)}
reps=${2:-3}
ms=${3:-3000}
read -ra builds <<< "${BUILDS:-mp}"
target=${TARGET:-1}

git submodule update --init --recursive

declare -A mode=([tds]=0 [mp]=1)
for build in "${builds[@]}"; do
    m=${mode[$build]:?unknown build $build (use mp or tds)}
    cmake -S . -B "build-bench-mp$m" -DCMAKE_BUILD_TYPE=Release -DNANA_GUI=OFF \
        -DCMAKE_CXX_FLAGS="-DNANA_MPMCTS=$m" > /dev/null
    cmake --build "build-bench-mp$m" --target nana_bench nana_msgbench -j "$(nproc)" > /dev/null
done
tds=build-bench-mp0/cpp/nana_bench
mp=build-bench-mp1/cpp/nana_bench

out="results/$(hostname)-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out"
{
    echo "host: $(hostname)  cpus: $(nproc)  threads: ${THREADS:-1..$max_threads}  reps: $reps  ms: $ms  builds: ${builds[*]}  target: $target"
    lscpu 2>/dev/null | grep -E "Model name|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|MHz" || true
    git log -1 --format="commit: %h %s"
} > "$out/machine.txt"

threads=()
if [ -n "${THREADS:-}" ]; then
    read -ra threads <<< "$THREADS"
else
    for ((n = 1; n < max_threads; n *= 2)); do threads+=("$n"); done
    threads+=("$max_threads")
fi

field() { grep "^$1" <<< "$2" | head -1 | awk -F': ' '{print $2}' | tr -d '%'; }

run() {
    local bin=${!1}
    local o wait
    o=$("$bin" "$3" "$5" "$2")
    wait=$(awk '/^ +[0-9]+ +[0-9.]+%/ { gsub("%", "", $2); s += $2; c++ } END { if (c) printf "%.1f", s / c }' <<< "$o")
    echo "$1,$2,$3,$4,$5,$(field 'simulations / second' "$o"),$(field 'nodes / second' "$o"),$wait,$(field 'tree depth' "$o"),$(field 'restarts' "$o"),$(field 'root backprops' "$o")" >> "$out/runs.csv"
    echo "test $2  threads $3  ms $5  rep $4  $1 done"
}

echo "build,test,threads,rep,ms,sims_per_s,nodes_per_s,wait_pct,depth,restarts,root_backprops" > "$out/runs.csv"
for test in 0 1; do
    for n in "${threads[@]}"; do
        for ((r = 1; r <= reps; r++)); do
            for build in "${builds[@]}"; do
                run "$build" "$test" "$n" "$r" "$ms"
                if [ "$target" = 1 ] && [ "$n" -gt 1 ]; then
                    run "$build" "$test" 1 "$r" $((n * ms))
                fi
            done
        done
    done
done

first=${builds[0]}
"$(dirname "${!first}")/nana_msgbench" scale "$max_threads" | tee "$out/scale.txt"

python3 - "$out/runs.csv" "$ms" <<'EOF' | tee "$out/summary.txt"
import csv, statistics as st, sys, collections
ms = int(sys.argv[2])
d = collections.defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    d[(r['build'], r['test'], int(r['threads']), int(r['ms']))].append(r)
med = lambda rs, k: st.median(float(x[k]) for x in rs if x[k])
builds = [b for b in ('tds', 'mp') if any(k[0] == b for k in d)]
for test in sorted({k[1] for k in d}):
    print(f"\ntest {test}: simulations/s, median of reps")
    print("  (speedup vs own 1 thread at the same ms; target = n x rate of 1 thread run n times as long)")
    for n in sorted({k[2] for k in d if k[1] == test and k[3] == ms}):
        cells = []
        for b in builds:
            rs = d[(b, test, n, ms)]
            s = med(rs, 'sims_per_s')
            cell = f"{b} {s / 1e3:8.1f}k ({s / med(d[(b, test, 1, ms)], 'sims_per_s'):5.2f}x"
            long = d.get((b, test, 1, n * ms))
            if long:
                t = n * med(long, 'sims_per_s')
                cell += f", target {t / 1e3:7.1f}k = {100 * s / t:3.0f}%"
            cells.append(cell + f") wait {med(rs, 'wait_pct'):5.1f}%")
        line = f"  {n:3d} threads | " + " | ".join(cells)
        if len(builds) == 2:
            line += f" | mp/tds x{med(d[('mp', test, n, ms)], 'sims_per_s') / med(d[('tds', test, n, ms)], 'sims_per_s'):.2f}"
        print(line)
EOF
echo "results in $out"
