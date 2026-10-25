#!/usr/bin/env bash
# Keep the full nana_bench output (including the per-thread profile) for a few runs.
# usage: scripts/profile_run.sh [threads] [reps] [ms_per_run] [build dir]
set -euo pipefail

cd "$(dirname "$0")/.."
threads=${1:-1,$(nproc)}
reps=${2:-1}
ms=${3:-3000}
build=${4:-build-bench-mp1}

if [ ! -x "$build/cpp/nana_bench" ]; then
    cmake -S . -B "$build" -DCMAKE_BUILD_TYPE=Release -DNANA_GUI=OFF > /dev/null
    cmake --build "$build" --target nana_bench -j "$(nproc)" > /dev/null
fi

out="results/profile-$(hostname)-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out"
git log -1 --format="commit: %h %s" > "$out/commit.txt"
git diff --stat >> "$out/commit.txt"

for n in ${threads//,/ }; do
    for test in 0 1; do
        for ((r = 1; r <= reps; r++)); do
            f="$out/t$n-test$test-rep$r.txt"
            "$build/cpp/nana_bench" "$n" "$ms" "$test" > "$f"
            echo "$f: $(grep '^simulations / second' "$f")"
        done
    done
done
echo "results in $out"
