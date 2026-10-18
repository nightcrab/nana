#!/usr/bin/env bash
# CPU profile of nana_bench on Linux: flame graphs at 1 thread and N threads, a
# differential graph between them, top functions by self time, and perf stat counters.
#
# usage: scripts/flamegraph.sh [threads] [ms_per_run] [test]
#   threads    the many-thread count to compare against 1 thread (default: nproc)
#
# Needs sudo (to install perf and relax perf_event_paranoid). Writes
# results/flame-<host>-<date>/ with *.svg, *.folded, top-*.txt and stat-*.txt.
set -euo pipefail

cd "$(dirname "$0")/.."
threads=${1:-$(nproc)}
ms=${2:-5000}
test=${3:-0}

# perf for the running kernel, and the FlameGraph scripts
if ! perf --version > /dev/null 2>&1; then
    sudo apt-get install -y -q "linux-tools-$(uname -r)" linux-tools-common > /dev/null
fi
[ -d FlameGraph ] || git clone -q --depth 1 https://github.com/brendangregg/FlameGraph.git
sudo sysctl -q kernel.perf_event_paranoid=1 kernel.kptr_restrict=0

# optimised, with symbols and frame pointers so call stacks are complete
build=build-flame
cmake -S . -B "$build" -DCMAKE_BUILD_TYPE=Release -DNANA_GUI=OFF \
    -DCMAKE_CXX_FLAGS="-g -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer" > /dev/null
cmake --build "$build" --target nana_bench -j "$(nproc)" > /dev/null
bench="$build/cpp/nana_bench"

out="results/flame-$(hostname)-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out"
git log -1 --format="commit: %h %s" > "$out/commit.txt"
git diff --stat >> "$out/commit.txt"

for n in 1 "$threads"; do
    echo "profiling $n thread(s)"
    perf record -q -F 499 -g --call-graph fp -o "$out/perf-t$n.data" -- "$bench" "$n" "$ms" "$test" > "$out/bench-t$n.txt"
    perf script -i "$out/perf-t$n.data" 2> /dev/null | FlameGraph/stackcollapse-perf.pl > "$out/t$n.folded"
    FlameGraph/flamegraph.pl --title "nana_bench, $n thread(s), test $test" "$out/t$n.folded" > "$out/t$n.svg"

    # top functions by self time (the leaf frame of each stack)
    awk '{ n = $NF; $NF = ""; k = split($0, f, ";"); leaf = f[k]; gsub(/ +$/, "", leaf); self[leaf] += n; total += n }
         END { for (s in self) printf "%6.2f%%  %s\n", 100 * self[s] / total, s }' "$out/t$n.folded" \
        | sort -rn | head -40 > "$out/top-self-t$n.txt"
    # and by inclusive time (any frame on the stack)
    awk '{ n = $NF; $NF = ""; k = split($0, f, ";"); delete seen; for (i = 1; i <= k; i++) { s = f[i]; gsub(/ +$/, "", s); if (!(s in seen)) { incl[s] += n; seen[s] = 1 } } total += n }
         END { for (s in incl) printf "%6.2f%%  %s\n", 100 * incl[s] / total, s }' "$out/t$n.folded" \
        | sort -rn | head -60 > "$out/top-inclusive-t$n.txt"

    # hardware counters, if the VM exposes them; software counters otherwise
    perf stat -e task-clock,context-switches,cpu-migrations,page-faults,cycles,instructions,cache-references,cache-misses,branch-misses \
        -o "$out/stat-t$n.txt" -- "$bench" "$n" "$ms" "$test" > /dev/null 2>&1 \
        || perf stat -o "$out/stat-t$n.txt" -- "$bench" "$n" "$ms" "$test" > /dev/null
done

# what grows from 1 thread to many: red = bigger share with more threads
FlameGraph/difffolded.pl -n "$out/t1.folded" "$out/t$threads.folded" \
    | FlameGraph/flamegraph.pl --title "1 -> $threads threads (red = grew)" > "$out/diff-1-to-$threads.svg"

# false sharing, if memory sampling is available in this VM
if perf c2c record -o "$out/c2c.data" -- "$bench" "$threads" 2000 "$test" > /dev/null 2>&1; then
    perf c2c report -i "$out/c2c.data" --stdio 2> /dev/null | head -200 > "$out/c2c.txt"
else
    echo "perf c2c not available on this machine" > "$out/c2c.txt"
fi

rm -f "$out"/*.data
echo "results in $out"
