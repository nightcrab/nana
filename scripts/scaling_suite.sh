#!/usr/bin/env bash
# Sim scaling, then strength scaling, back to back on one machine.
# Meant for scripts/run_ec2_bench.sh SCRIPT=scaling_suite.sh.
#
#   1. speed_scaling.sh 32 5 3000 (needs the machine to itself, so it goes first)
#   2. a 16-game strength smoke run all the way to its summary, so a broken build fails in minutes
#   3. strength_batches.sh at the size of the 2026-10-20 run: 3 x 100 games, 1-thread ladder
#      x1..x32 at 80 ms, 2/4/8/16/32 threads at 80 ms, plus LF1
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== speed scaling ($(date +%H:%M:%S))"
bash scripts/speed_scaling.sh 32 5 3000

strength=(BASE_MS=80 MULTS="1 2 4 8 16 32" PAR_THREADS="2 4 8 16 32" LF1=1 HORIZON=50)

echo "=== strength smoke ($(date +%H:%M:%S))"
env "${strength[@]}" MULTS=1 PAR_THREADS=32 GAMES=16 SEED0=900000 BUILD=1 bash scripts/strength_ladder.sh
smoke=$(ls -d results/*-strength-* | sort | tail -n 1)
python3 scripts/strength_report.py "$smoke" > /dev/null  # the ladder only warns if the report fails
mv "$smoke" "${smoke/-strength-/-smoke-}"  # keep it out of the batches' bookkeeping

echo "=== strength scaling ($(date +%H:%M:%S))"
bash scripts/strength_batches.sh "${strength[@]}" BATCHES=3 GAMES=100 FIRST_SEED=0 BUILD=0
echo "=== done ($(date +%H:%M:%S))"
