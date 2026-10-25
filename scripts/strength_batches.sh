#!/usr/bin/env bash
# Run strength_ladder.sh in batches of fresh seeds.
# usage: scripts/strength_batches.sh [NAME=value ...]
set -euo pipefail

cd "$(dirname "$0")/.."
for setting in "$@"; do
    if [[ $setting != [A-Z]*=* ]]; then
        echo "expected NAME=value, got $setting" >&2
        exit 1
    fi
    export "${setting?}"
done
batches=${BATCHES:-3}
games=${GAMES:-100}
first_seed=${FIRST_SEED:-0}

mkdir -p results
for ((b = 0; b < batches; b++)); do
    seed0=$(( first_seed + b * games ))
    echo "=== batch $(( b + 1 )) of $batches: seeds $seed0..$(( seed0 + games - 1 )) ($(date +%H:%M:%S))"
    before=$(ls -d results/*-strength-* 2> /dev/null | sort || true)
    BUILD=$(( b == 0 ? ${BUILD:-1} : 0 )) GAMES=$games SEED0=$seed0 bash scripts/strength_ladder.sh
    dir=$(comm -13 <(echo "$before") <(ls -d results/*-strength-* | sort) | tail -n 1)
    echo "$dir seeds $seed0..$(( seed0 + games - 1 ))" >> results/batches.txt
    echo "=== batch $(( b + 1 )) done: $dir"
done
