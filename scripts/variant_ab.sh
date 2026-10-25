#!/usr/bin/env bash
# Paired A/B of strength variants at a fixed simulation budget.
# usage: scripts/variant_ab.sh [NAME=value ...]
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
