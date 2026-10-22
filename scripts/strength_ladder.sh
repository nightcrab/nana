#!/usr/bin/env bash
# Playing-strength scaling benchmark for MP-MCTS: a 1-thread think-time ladder against
# multi-thread runs at the base think time, all on the same seeds.
#
# Garbage comes from the emulated opponent at its normal rate, with no ramp.
#
# Every game is played for a fixed horizon of HORIZON pieces (or until the bot dies), and
# strength is attack per piece over that horizon: attack / HORIZON, so a bot that dies early
# sends nothing for the pieces it didn't play. That matches what the search maximises (attack
# per piece over its lookahead, with death worth 0), gives every config the same cost per game,
# and scores everyone on the same pieces, so stronger bots aren't judged on later, harder ones.
#
# usage: scripts/strength_ladder.sh [NAME=value ...]
# settings, as env vars or NAME=value arguments (which is how scripts/run_ec2_bench.sh passes
# them, e.g. SCRIPT=strength_ladder.sh run_ec2_bench.sh <host> <key> master SEED0=100);
# defaults in brackets:
#   BASE_MS      base think time per move [80]
#   MULTS        think-time multiples for the 1-thread ladder, space or comma separated, or none
#                (to pool a ladder from an earlier run on the same seeds) ["1 2 4 8 16"]
#   PAR_THREADS  thread counts played at the base think time ["16"]; each must divide nproc,
#                and nproc / threads of them run at once (16 on 32 cpus: two at a time)
#   GAMES        games per config, on seeds SEED0..SEED0+GAMES-1 [300]
#   SEED0        first seed; use a fresh range (300, 600, ...) for a rerun you want to pool [0]
#   HORIZON      pieces per game [50]
#   LF1          1 = also play 1 thread at the base think time with LOAD_FACTOR 1 [1]
#   CHUNK_UNITS  base-think-time games per task, so long configs are split finely [8]
#   EXTRA_FLAGS  compile flags for the plain builds (ladder, LF1, and parallel configs without
#                VARIANTS), comma separated, e.g. -DNANA_LEAK=3,-DNANA_LEAK_EPS=0.1 [""]
#   VARIANTS     builds to compare, comma separated names from scripts/strength_variants.txt;
#                each parallel config is played once per variant (named t<n>-x1-lf6-v<name>),
#                interleaved task by task on the same seeds. empty = the plain build [""]
#   BUILD        0 = don't build: reuse earlier builds, or BIN_LF6 / BIN_LF1 if set (BIN_LF6 then
#                stands in for every variant) [1]
#   PIN          0 = don't taskset (for a plumbing check off Linux) [1]
#
# Configs are named t<threads>-x<multiple>-lf<load factor>[-v<variant>]. The 1-thread configs run nproc at
# a time, one per cpu: each game then shares its L3 with as many others as a worker does
# inside the full-width search, so the comparison holds the hardware fixed. An N-thread
# config runs nproc / N games at a time on consecutive cpu blocks (consecutive cpus share an
# L3 on AWS c7a; no SMT assumed). Tasks go to slots through a shared queue, longest first,
# so the run doesn't end on one slot still playing 16x games. A task that runs past twice its
# full-horizon time is killed and reported, so a hung search can't stall the run.
#
# Writes results/<host>-strength-<date>/{machine.txt,<config>.jsonl,summary.txt}, plus each
# task's stderr in parts/; each game line gets config, mult, base_ms and load_factor fields.
# Then: python3 scripts/strength_report.py <dir> [--plot out.png]
# Needs cmake, a C++23 compiler (gcc 13+ or clang 17+), taskset, timeout and python3.
set -euo pipefail

cd "$(dirname "$0")/.."
for setting in "$@"; do
    if [[ $setting != [A-Z]*=* ]]; then
        echo "expected NAME=value, got $setting" >&2
        exit 1
    fi
    export "${setting?}"
done
base_ms=${BASE_MS:-80}
mults_setting=${MULTS:-1 2 4 8 16}
[ "$mults_setting" = none ] && mults_setting=""
read -ra mults <<< "$(tr , " " <<< "$mults_setting")"
read -ra variants <<< "$(tr , " " <<< "${VARIANTS:-}")"
extra_flags=$(tr , " " <<< "${EXTRA_FLAGS:-}")
variants_file=scripts/strength_variants.txt
variant_flags() {
    awk -v name="$1" '$1 == name { $1 = ""; sub(/^ +/, ""); print; found = 1 } END { exit !found }' "$variants_file"
}
for v in "${variants[@]}"; do
    variant_flags "$v" > /dev/null || { echo "no variant $v in $variants_file" >&2; exit 1; }
done
read -ra par_threads <<< "$(tr , " " <<< "${PAR_THREADS:-16}")"
games=${GAMES:-300}
seed0=${SEED0:-0}
horizon=${HORIZON:-50}
lf1=${LF1:-1}
chunk_units=${CHUNK_UNITS:-8}
pin=${PIN:-1}
cpus=$(nproc)

for n in "${par_threads[@]}"; do
    if (( n < 1 || n > cpus || cpus % n )); then
        echo "PAR_THREADS: $n doesn't divide $cpus cpus" >&2
        exit 1
    fi
done

if [ "${BUILD:-1}" = 1 ]; then
    git submodule update --init --recursive
    for lf in 6 1; do
        [ "$lf" = 1 ] && [ "$lf1" != 1 ] && continue
        # the plain build plays the ladder and, without variants, the parallel configs
        [ "$lf" = 6 ] && (( ${#mults[@]} == 0 && ${#variants[@]} > 0 )) && continue
        cmake -S . -B "build-strength-lf$lf" -DCMAKE_BUILD_TYPE=Release -DNANA_GUI=OFF \
            -DCMAKE_CXX_FLAGS="-DNANA_MPMCTS=1 -DNANA_LOAD_FACTOR=$lf $extra_flags" > /dev/null
        cmake --build "build-strength-lf$lf" --target nana_emulate -j "$cpus" > /dev/null
    done
    for v in "${variants[@]}"; do
        cmake -S . -B "build-strength-v$v" -DCMAKE_BUILD_TYPE=Release -DNANA_GUI=OFF             -DCMAKE_CXX_FLAGS="-DNANA_MPMCTS=1 -DNANA_LOAD_FACTOR=6 $(variant_flags "$v")" > /dev/null
        cmake --build "build-strength-v$v" --target nana_emulate -j "$cpus" > /dev/null
    done
fi
# a task names its binary by key: 6 and 1 are the plain builds by load factor, v<name> a variant
declare -A bins=([6]=${BIN_LF6:-$PWD/build-strength-lf6/cpp/nana_emulate} [1]=${BIN_LF1:-$PWD/build-strength-lf1/cpp/nana_emulate})
# with BUILD=0 (later batches of strength_batches.sh) a variant reuses its earlier build, unless
# BIN_LF6 is set, which then stands in for every variant (a plumbing check off Linux)
for v in "${variants[@]}"; do
    bins[v$v]=${BIN_LF6:-$PWD/build-strength-v$v/cpp/nana_emulate}
done
for key in "${!bins[@]}"; do
    [[ $key == v* ]] && [ ! -x "${bins[$key]}" ] && { echo "missing binary for variant ${key#v}: ${bins[$key]}" >&2; exit 1; }
done

# the APP benchmark measures our own attack: score rollouts by APP and eval only, not the opponent's death
export NANA_REWARD=${NANA_REWARD:-app}

out="results/$(hostname)-strength-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out/parts"
log() { echo "$*" | tee -a "$out/machine.txt"; }
{
    echo "host: $(hostname)  cpus: $cpus  base_ms: $base_ms  mults: ${mults[*]}  par_threads: ${par_threads[*]}"
    echo "games: $games  seeds: $seed0..$(( seed0 + games - 1 ))  horizon: $horizon  lf1: $lf1"
    echo "extra flags (plain builds): ${extra_flags:-none}"
    for v in "${variants[@]}"; do echo "variant $v: $(variant_flags "$v")"; done
    lscpu 2>/dev/null | grep -E "Model name|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|L3|MHz" || true
    uname -r
    git log -1 --format="commit: %h %s"
    echo "uncommitted:"
    git status --short
    env | grep '^NANA_' || true
} > "$out/machine.txt"

# a freshly booted Ubuntu runs apt in its first minutes, on the cpus we're about to fill
if sudo -n true 2> /dev/null; then
    for unit in apt-daily.timer apt-daily-upgrade.timer unattended-upgrades.service snapd.service snapd.socket; do
        sudo systemctl stop "$unit" 2> /dev/null || true
    done
    for ((i = 0; i < 60; i++)); do
        pgrep -x 'apt|apt-get|dpkg|unattended-upgr' > /dev/null || break
        sleep 5
    done
fi
log "load average at start: $(cut -d' ' -f1-3 /proc/loadavg 2> /dev/null)"

# run_tasks <threads> <dir> <task>...: plays every task, cpus / threads at a time. a task is
# "<name> <bin> <ms> <seed> <games>", <bin> a key of bins; its games go to <dir>/<name>-seed<seed>.jsonl
run_tasks() {
    local threads=$1 dir=$2
    shift 2
    local tasks=("$@") slots=$(( cpus / threads )) claims
    claims=$(mktemp -d)
    worker() {
        local slot=$1 first=$(( $1 * threads )) k name lf ms seed n limit status
        local pinning=()
        [ "$pin" = 1 ] && pinning=(taskset -c "$first-$(( first + threads - 1 ))")
        for ((k = 0; k < ${#tasks[@]}; k++)); do
            # claim task k: bash's noclobber create is O_EXCL, so exactly one slot wins. (not mkdir:
            # Ubuntu 26.04's Rust coreutils mkdir let several slots claim one task, which then wrote
            # the same files at once)
            ( set -C; : > "$claims/$k" ) 2> /dev/null || continue
            read -r name lf ms seed n <<< "${tasks[$k]}"
            limit=$(( 2 * n * horizon * ms / 1000 + 60 ))
            status=0
            timeout "$limit" "${pinning[@]}" "${bins[$lf]}" --threads "$threads" --games "$n" --ms "$ms" \
                --seed "$seed" --max-pieces "$horizon" \
                > "$dir/$name-seed$seed.jsonl" 2> "$dir/$name-seed$seed.err" || status=$?
            if [ "$status" = 124 ]; then
                echo "  FAILED: $name seeds $seed..$(( seed + n - 1 )) timed out after ${limit}s"
            elif [ "$status" != 0 ]; then
                # 128 + n: killed by signal n (9 = SIGKILL, e.g. out of memory; 11 = SIGSEGV)
                echo "  FAILED: $name seeds $seed..$(( seed + n - 1 )) exit $status$( (( status > 128 )) && echo " (signal $(( status - 128 )))")"
            fi
            echo "  $(date +%H:%M:%S) $name seeds $seed..$(( seed + n - 1 )) done ($(( k + 1 ))/${#tasks[@]} started)"
        done
    }
    for ((slot = 0; slot < slots; slot++)); do
        worker "$slot" &
    done
    wait
    rm -rf "$claims"
}

# chunk <name> <lf> <mult> <first seed> <games>: tasks of about chunk_units base games
chunk() {
    local name=$1 lf=$2 m=$3 first=$4 count=$5 size seed
    size=$(( chunk_units / m > 0 ? chunk_units / m : 1 ))
    for ((seed = first; seed < first + count; seed += size)); do
        echo "$name $lf $(( base_ms * m )) $seed $(( first + count - seed < size ? first + count - seed : size ))"
    done
}

ladder_units=0
for m in "${mults[@]}"; do ladder_units=$(( ladder_units + m )); done
# an N-thread config plays cpus / N games at a time, so it costs N cpus per game
par_units=0
for n in "${par_threads[@]}"; do par_units=$(( par_units + n * (${#variants[@]} > 0 ? ${#variants[@]} : 1) )); done
log "estimate: 1-thread ladder $(( (ladder_units + lf1) * games * base_ms * horizon / 1000 / cpus ))s," \
    "parallel runs $(( par_units * games * base_ms * horizon / 1000 / cpus ))s (less where games die early)"

for n in "${par_threads[@]}"; do
    log "$(date +%H:%M:%S) $n threads at ${base_ms}ms, $(( cpus / n )) at a time"
    if (( ${#variants[@]} == 0 )); then
        mapfile -t tasks < <(chunk "t$n-x1-lf6" 6 1 "$seed0" "$games")
    else
        # every variant's chunk of seeds before the next chunk, so drift hits them all alike
        tasks=()
        while read -r _ _ ms seed count; do
            for v in "${variants[@]}"; do
                tasks+=("t$n-x1-lf6-v$v v$v $ms $seed $count")
            done
        done < <(chunk "t$n-x1-lf6" 6 1 "$seed0" "$games")
    fi
    run_tasks "$n" "$out/parts" "${tasks[@]}"
done

(( ${#mults[@]} > 0 || lf1 == 1 )) && log "$(date +%H:%M:%S) 1-thread ladder, $cpus at a time"
tasks=()
for ((i = ${#mults[@]} - 1; i >= 0; i--)); do
    mapfile -t -O "${#tasks[@]}" tasks < <(chunk "t1-x${mults[i]}-lf6" 6 "${mults[i]}" "$seed0" "$games")
done
if [ "$lf1" = 1 ]; then
    mapfile -t -O "${#tasks[@]}" tasks < <(chunk "t1-x1-lf1" 1 1 "$seed0" "$games")
fi
(( ${#tasks[@]} > 0 )) && run_tasks 1 "$out/parts" "${tasks[@]}"
log "$(date +%H:%M:%S) done; load average at end: $(cut -d' ' -f1-3 /proc/loadavg 2> /dev/null)"

python3 scripts/strength_merge.py "$out" "$base_ms"
# the games are saved by now: a report error shouldn't fail the run
python3 scripts/strength_report.py "$out" | tee "$out/summary.txt" || echo "report failed; rerun strength_report.py on $out"
echo "results in $out"
