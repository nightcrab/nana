#!/usr/bin/env bash
# Playing-strength benchmark: a 1-thread think-time ladder against multi-thread runs.
# usage: scripts/strength_ladder.sh [NAME=value ...]
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
sims=${SIMS:-0}
mults_setting=${MULTS:-1 2 4 8 16}
[ "$mults_setting" = none ] && mults_setting=""
read -ra mults <<< "$(tr , " " <<< "$mults_setting")"
read -ra variants <<< "$(tr , " " <<< "${VARIANTS:-}")"
extra_flags=$(tr , " " <<< "${EXTRA_FLAGS:-}")
variants_file=scripts/strength_variants.txt
variant_flags() {
    awk -v name="$1" '{ sub(/\r$/, "") } $1 == name { $1 = ""; sub(/^ +/, ""); print; found = 1 } END { exit !found }' "$variants_file"
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
        [ "$lf" = 6 ] && (( ${#mults[@]} == 0 && ${#variants[@]} > 0 )) && continue
        cmake -S . -B "build-strength-lf$lf" -DCMAKE_BUILD_TYPE=Release -DNANA_GUI=OFF \
            -DCMAKE_CXX_FLAGS="-DNANA_LOAD_FACTOR=$lf $extra_flags" > /dev/null
        cmake --build "build-strength-lf$lf" --target nana_emulate -j "$cpus" > /dev/null
    done
    for v in "${variants[@]}"; do
        cmake -S . -B "build-strength-v$v" -DCMAKE_BUILD_TYPE=Release -DNANA_GUI=OFF             -DCMAKE_CXX_FLAGS="-DNANA_LOAD_FACTOR=6 $(variant_flags "$v")" > /dev/null
        cmake --build "build-strength-v$v" --target nana_emulate -j "$cpus" > /dev/null
    done
fi
declare -A bins=([6]=${BIN_LF6:-$PWD/build-strength-lf6/cpp/nana_emulate} [1]=${BIN_LF1:-$PWD/build-strength-lf1/cpp/nana_emulate})
for v in "${variants[@]}"; do
    bins[v$v]=${BIN_LF6:-$PWD/build-strength-v$v/cpp/nana_emulate}
done
for key in "${!bins[@]}"; do
    [[ $key == v* ]] && [ ! -x "${bins[$key]}" ] && { echo "missing binary for variant ${key#v}: ${bins[$key]}" >&2; exit 1; }
done

out="results/$(hostname)-strength-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out/parts"
# Runtime variants inherit the supplied config, then override only their own settings.
declare -A variant_configs
for v in "${variants[@]}"; do
    if [ "$v" = no-garbage ]; then
        variant_configs[v$v]="$PWD/$out/config-$v.json"
        python3 - "${NANA_CONFIG:-}" "${variant_configs[v$v]}" <<'PY'
import json
import sys

config = {}
if sys.argv[1]:
    with open(sys.argv[1], encoding="utf-8") as source:
        config = json.load(source)
config.update(opponent=False, belief_weights=[0] * 8)
with open(sys.argv[2], "w", encoding="utf-8") as target:
    json.dump(config, target, indent=2)
    target.write("\n")
PY
    fi
done
log() { echo "$*" | tee -a "$out/machine.txt"; }
{
    echo "host: $(hostname)  cpus: $cpus  base_ms: $base_ms  mults: ${mults[*]}  par_threads: ${par_threads[*]}"
    echo "games: $games  seeds: $seed0..$(( seed0 + games - 1 ))  horizon: $horizon  lf1: $lf1"
    (( sims > 0 )) && echo "fixed search: $sims simulations per move (ms columns are only BASE_MS)"
    echo "extra flags (plain builds): ${extra_flags:-none}"
    for v in "${variants[@]}"; do
        echo "variant $v: $(variant_flags "$v")"
        if [ -n "${variant_configs[v$v]:-}" ]; then
            echo "  NANA_CONFIG=${variant_configs[v$v]}"
            cat "${variant_configs[v$v]}"
        fi
    done
    lscpu 2>/dev/null | grep -E "Model name|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|L3|MHz" || true
    uname -r
    git log -1 --format="commit: %h %s"
    echo "uncommitted:"
    git status --short
    env | grep '^NANA_' || true
} > "$out/machine.txt"

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

run_tasks() {
    local threads=$1 dir=$2
    shift 2
    local tasks=("$@") slots=$(( cpus / threads )) claims
    claims=$(mktemp -d)
    worker() {
        local slot=$1 first=$(( $1 * threads )) k name lf ms seed n limit status
        local pinning=() budget=() runtime=()
        [ "$pin" = 1 ] && pinning=(taskset -c "$first-$(( first + threads - 1 ))")
        (( sims > 0 )) && budget=(--sims "$sims")
        for ((k = 0; k < ${#tasks[@]}; k++)); do
            ( set -C; : > "$claims/$k" ) 2> /dev/null || continue
            read -r name lf ms seed n <<< "${tasks[$k]}"
            runtime=()
            if [ -n "${variant_configs[$lf]:-}" ]; then
                runtime=(env "NANA_CONFIG=${variant_configs[$lf]}")
            fi
            limit=$(( (sims > 0 ? 6 : 2) * n * horizon * ms / 1000 + 60 ))
            status=0
            timeout "$limit" "${pinning[@]}" "${runtime[@]}" "${bins[$lf]}" --threads "$threads" --games "$n" --ms "$ms" \
                --seed "$seed" --max-pieces "$horizon" "${budget[@]}" \
                > "$dir/$name-seed$seed.jsonl" 2> "$dir/$name-seed$seed.err" || status=$?
            if [ "$status" = 124 ]; then
                echo "  FAILED: $name seeds $seed..$(( seed + n - 1 )) timed out after ${limit}s"
            elif [ "$status" != 0 ]; then
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

chunk() {
    local name=$1 lf=$2 m=$3 first=$4 count=$5 size seed
    size=$(( chunk_units / m > 0 ? chunk_units / m : 1 ))
    for ((seed = first; seed < first + count; seed += size)); do
        echo "$name $lf $(( base_ms * m )) $seed $(( first + count - seed < size ? first + count - seed : size ))"
    done
}

ladder_units=0
for m in "${mults[@]}"; do ladder_units=$(( ladder_units + m )); done
par_units=0
for n in "${par_threads[@]}"; do par_units=$(( par_units + n * (${#variants[@]} > 0 ? ${#variants[@]} : 1) )); done
log "estimate: 1-thread ladder $(( (ladder_units + lf1) * games * base_ms * horizon / 1000 / cpus ))s," \
    "parallel runs $(( par_units * games * base_ms * horizon / 1000 / cpus ))s (less where games die early)"

for n in "${par_threads[@]}"; do
    log "$(date +%H:%M:%S) $n threads at ${base_ms}ms, $(( cpus / n )) at a time"
    if (( ${#variants[@]} == 0 )); then
        mapfile -t tasks < <(chunk "t$n-x1-lf6" 6 1 "$seed0" "$games")
    else
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
python3 scripts/strength_report.py "$out" | tee "$out/summary.txt" || echo "report failed; rerun strength_report.py on $out"
echo "results in $out"
