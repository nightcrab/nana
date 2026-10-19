#!/usr/bin/env bash
# Playing-strength benchmark: survival time of TDS-UCT vs MP-MCTS at a ladder of think times.
#
# usage: scripts/survival_ladder.sh [threads] [games] [ms_list] [ramp] [chunk]
#   threads  workers per game (default 8, one chiplet on the 32-core EC2 box)
#   games    games per (build, think time), on seeds 0..games-1 (default 100)
#   ms_list  think times per move, comma separated (default 10,25,50,100)
#   ramp     nana_play --ramp: pieces until incoming garbage doubles (default 0 = off)
#   chunk    games per nana_play run (default 10)
#
# Builds nana_play twice (NANA_MPMCTS=0 and 1, headless, Release). Runs nproc / threads
# games at once, each pinned with taskset to its own block of cpus; that assumes no SMT
# and that consecutive cpus share an L3 (true on AWS c7a). Both builds play the same
# seeds, interleaved, so drift hits both equally and they can be compared seed by seed.
#
# Writes results/<host>-<date>-survival/{<build>-ms<ms>.jsonl,summary.txt}.
# Needs cmake, a C++23 compiler (gcc 13+ or clang 17+), taskset and python3.
set -euo pipefail

cd "$(dirname "$0")/.."
threads=${1:-8}
games=${2:-100}
IFS=, read -ra ms_list <<< "${3:-10,25,50,100}"
ramp=${4:-0}
chunk=${5:-10}

slots=$(( $(nproc) / threads ))
if (( slots < 1 )); then
    echo "need at least $threads cpus" >&2
    exit 1
fi

git submodule update --init --recursive

for mode in 0 1; do
    cmake -S . -B "build-play-mp$mode" -DCMAKE_BUILD_TYPE=Release -DNANA_GUI=OFF \
        -DCMAKE_CXX_FLAGS="-DNANA_MPMCTS=$mode" > /dev/null
    cmake --build "build-play-mp$mode" --target nana_play -j "$(nproc)" > /dev/null
done
declare -A bins=([tds]=build-play-mp0/cpp/nana_play [mp]=build-play-mp1/cpp/nana_play)

out="results/$(hostname)-$(date +%Y%m%d-%H%M%S)-survival"
mkdir -p "$out/parts"
{
    echo "host: $(hostname)  cpus: $(nproc)  threads: $threads  slots: $slots  games: $games"
    echo "ms: ${ms_list[*]}  ramp: $ramp  chunk: $chunk"
    lscpu 2>/dev/null | grep -E "Model name|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|MHz" || true
    git log -1 --format="commit: %h %s"
    env | grep '^NANA_' || true
} > "$out/machine.txt"

# tasks interleave builds and think times; task k goes to slot k % slots
tasks=()
for ((seed = 0; seed < games; seed += chunk)); do
    n=$(( games - seed < chunk ? games - seed : chunk ))
    for ms in "${ms_list[@]}"; do
        for build in tds mp; do
            tasks+=("$build $ms $seed $n")
        done
    done
done

run_slot() {
    local slot=$1 first=$(( $1 * threads )) last=$(( $1 * threads + threads - 1 ))
    for ((k = slot; k < ${#tasks[@]}; k += slots)); do
        read -r build ms seed n <<< "${tasks[$k]}"
        taskset -c "$first-$last" "${bins[$build]}" --threads "$threads" --games "$n" --ms "$ms" \
            --seed "$seed" --ramp "$ramp" > "$out/parts/$build-ms$ms-seed$seed.jsonl" 2> /dev/null
        echo "slot $slot: $build ms $ms seeds $seed..$(( seed + n - 1 )) done"
    done
}

for ((slot = 0; slot < slots; slot++)); do
    run_slot "$slot" &
done
wait

for ms in "${ms_list[@]}"; do
    for build in tds mp; do
        cat "$out"/parts/"$build-ms$ms"-seed*.jsonl > "$out/$build-ms$ms.jsonl"
    done
done

{
    for ms in "${ms_list[@]}"; do
        echo
        python3 scripts/survival_summary.py --paired "$out/tds-ms$ms.jsonl" "$out/mp-ms$ms.jsonl"
    done
} | tee "$out/summary.txt"
echo "results in $out"
