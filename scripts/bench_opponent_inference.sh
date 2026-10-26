#!/usr/bin/env bash
# Standalone, bounded inference benchmark. No repo checkout or training dependencies.
set -euo pipefail
ulimit -v 2097152
cd "$(dirname "$0")/.."
mkdir -p measured
lscpu > measured/lscpu.txt
lscpu -e > measured/topology.txt
free -m > measured/memory.txt
ps -eo pid,comm,pcpu,pmem --sort=-pcpu > measured/load-before.txt
g++ --version > measured/compiler.txt
: > measured/results.jsonl
for model in results/opponent-infer-w*; do
    name=$(basename "$model")
    for isa in avx512 avx2; do
        flags=(-O3 -std=c++20 -march=znver5 -pthread)
        if [[ $isa == avx2 ]]; then flags+=(-mno-avx512f); fi
        printf '%q ' g++ "${flags[@]}" -I "$model" cpp/opponent_inference/bench.cpp -o "measured/$name-$isa" >> measured/build-commands.txt
        printf '\n' >> measured/build-commands.txt
        /usr/bin/time -v -o "measured/$name-$isa-build-time.txt" g++ "${flags[@]}" -I "$model" cpp/opponent_inference/bench.cpp -o "measured/$name-$isa"
        for threads in 1 8 16 32; do
            log="measured/$name-$isa-t$threads.log"
            /usr/bin/time -v -o "$log.time" "measured/$name-$isa" "$model" "$threads" 500 32 50 1 1 > "$log"
            python3 -c 'import json,sys; row=json.loads(open(sys.argv[1]).readlines()[-1]); row.update(model=sys.argv[2],isa=sys.argv[3]); print(json.dumps(row))' "$log" "$name" "$isa" >> measured/results.jsonl
            tail -1 measured/results.jsonl
        done
    done
    # Compare blocking versus spinning barriers at the same model size/workload.
    log="measured/$name-avx512-t16-blocking.log"
    /usr/bin/time -v -o "$log.time" "measured/$name-avx512" "$model" 16 500 32 50 1 0 > "$log"
    python3 -c 'import json,sys; row=json.loads(open(sys.argv[1]).readlines()[-1]); row.update(model=sys.argv[2],isa="avx512"); print(json.dumps(row))' "$log" "$name" >> measured/results.jsonl
    tail -1 measured/results.jsonl
done
ps -eo pid,comm,pcpu,pmem --sort=-pcpu > measured/load-after.txt
