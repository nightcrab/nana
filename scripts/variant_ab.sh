#!/usr/bin/env bash
# Paired A/B of strength variants at a fixed simulation budget, 1 thread per game: a short smoke
# run all the way to its summary first (so a broken variant fails in minutes), then the full run.
# Meant for scripts/run_ec2_bench.sh SCRIPT=variant_ab.sh.
#
# usage: scripts/variant_ab.sh [NAME=value ...]
#   VARIANTS  names from scripts/strength_variants.txt, comma separated (required)
#   SIMS      simulations per move [4000]
#   GAMES     games per variant in the full run [1000]
#   SEED0     first seed of the full run [1000]
set -euo pipefail

cd "$(dirname "$0")/.."
for setting in "$@"; do
    export "${setting?}"
done
: "${VARIANTS:?set VARIANTS}"
common=(MULTS=none LF1=0 PAR_THREADS=1 SIMS="${SIMS:-4000}" BASE_MS=90 VARIANTS="$VARIANTS")

echo "=== smoke ($(date +%H:%M:%S))"
env "${common[@]}" GAMES=16 SEED0=0 BUILD=1 bash scripts/strength_ladder.sh
echo "=== full ($(date +%H:%M:%S))"
env "${common[@]}" GAMES="${GAMES:-1000}" SEED0="${SEED0:-1000}" BUILD=0 bash scripts/strength_ladder.sh
echo "=== done ($(date +%H:%M:%S))"
