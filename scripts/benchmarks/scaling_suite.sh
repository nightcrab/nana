#!/usr/bin/env bash
# Sim scaling, then strength scaling, back to back on one machine.
set -euo pipefail

cd "$(dirname "$0")/../.."

echo "=== speed scaling ($(date +%H:%M:%S))"
bash scripts/benchmarks/speed_scaling.sh 32 5 3000

strength=(BASE_MS=80 MULTS="1 2 4 8 16 32" PAR_THREADS="2 4 8 16 32" LF1=1 HORIZON=50)

echo "=== strength smoke ($(date +%H:%M:%S))"
env "${strength[@]}" MULTS=1 PAR_THREADS=32 GAMES=16 SEED0=900000 BUILD=1 bash scripts/benchmarks/strength_ladder.sh
smoke=$(ls -d results/*-strength-* | sort | tail -n 1)
python3 scripts/benchmarks/strength_report.py "$smoke" > /dev/null  # the ladder only warns if the report fails
mv "$smoke" "${smoke/-strength-/-smoke-}"  # keep it out of the batches' bookkeeping

echo "=== strength scaling ($(date +%H:%M:%S))"
bash scripts/benchmarks/strength_batches.sh "${strength[@]}" BATCHES=3 GAMES=100 FIRST_SEED=0 BUILD=0
echo "=== done ($(date +%H:%M:%S))"
