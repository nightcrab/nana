#!/usr/bin/env bash
# Simulations / second at every thread count from 1 to max_threads.
# usage: scripts/speed_scaling.sh [max_threads] [reps] [ms_per_run]
set -euo pipefail

cd "$(dirname "$0")/.."
max_threads=${1:-$(nproc)}
reps=${2:-5}
ms=${3:-3000}
read -ra tests <<< "${TESTS:-0 1}"
prep=${PREP:-1}

git submodule update --init --recursive
cmake -S . -B build-speed -DCMAKE_BUILD_TYPE=Release -DNANA_GUI=OFF > /dev/null
cmake --build build-speed --target nana_speed -j "$(nproc)" > /dev/null
bench=$PWD/build-speed/cpp/nana_speed

out="results/$(hostname)-speed-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out"
log() { echo "$*" | tee -a "$out/machine.txt"; }

{
    echo "host: $(hostname)  cpus: $(nproc)  threads: 1..$max_threads  reps: $reps  ms: $ms  tests: ${tests[*]}"
    lscpu 2>/dev/null | grep -E "Model name|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)|L3|MHz" || true
    uname -r
    git log -1 --format="commit: %h %s"
    echo "uncommitted:"
    git status --short
} > "$out/machine.txt"

nice=()
if [ "$prep" = 1 ] && sudo -n true 2> /dev/null; then
    log "prep: stopping background services"
    for unit in apt-daily.timer apt-daily-upgrade.timer unattended-upgrades.service snapd.service snapd.socket \
                man-db.timer fwupd-refresh.timer motd-news.timer; do
        sudo systemctl stop "$unit" 2> /dev/null && log "  stopped $unit" || true
    done
    for ((i = 0; i < 60; i++)); do
        pgrep -x 'apt|apt-get|dpkg|unattended-upgr' > /dev/null || break
        [ "$i" = 0 ] && log "  waiting for apt/dpkg to finish"
        sleep 5
    done
    if ls /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor > /dev/null 2>&1; then
        echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null \
            && log "prep: governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
    else
        log "prep: no cpufreq governor exposed (frequency is the hypervisor's)"
    fi
    if [ -f /proc/sys/kernel/numa_balancing ]; then
        sudo sysctl -q kernel.numa_balancing=0 && log "prep: numa_balancing 0"
    fi
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
    nice=(sudo -n nice -n -20)
    log "prep: running at nice -20"
else
    log "prep: skipped (PREP=0 or no passwordless sudo)"
fi
log "transparent hugepages: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2> /dev/null)"

for ((i = 0; i < 36; i++)); do
    awk '{ exit !($1 < 0.5) }' /proc/loadavg && break
    sleep 5
done
log "load average at start: $(cut -d' ' -f1-3 /proc/loadavg)"

sims() { "${nice[@]}" "$bench" "$1" "$ms" "$2" | awk -F': ' '/^simulations \/ second/ { print $2 }'; }

for test in "${tests[@]}"; do
    sims "$max_threads" "$test" > /dev/null
done

echo "test,threads,rep,order,sims_per_s" > "$out/runs.csv"
for ((r = 1; r <= reps; r++)); do
    k=0
    for n in $(seq 1 "$max_threads" | shuf --random-source=<(yes "$r")); do
        k=$((k + 1))
        for test in "${tests[@]}"; do
            s=$(sims "$n" "$test")  # a crashed run stops the sweep (set -e)
            : "${s:?no simulations / second from $n threads, test $test}"
            echo "$test,$n,$r,$k,$s" >> "$out/runs.csv"
        done
    done
    echo "rep $r of $reps done"
done
log "load average at end: $(cut -d' ' -f1-3 /proc/loadavg)"

python3 - "$out/runs.csv" <<'EOF' | tee "$out/summary.txt"
import collections, csv, statistics as st, sys
d = collections.defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    d[(r["test"], int(r["threads"]))].append(float(r["sims_per_s"]))
print("simulations / second, median of reps; speedup and efficiency vs the 1-thread median;")
print("spread = (max - min) / median over reps")
for test in sorted({t for t, _ in d}):
    base = st.median(d[(test, 1)])
    print(f"\ntest {test}")
    print(f"  {'thr':>3} {'sims/s':>10} {'speedup':>8} {'eff':>5} {'spread':>7} {'min':>10} {'max':>10}")
    for n in sorted(n for t, n in d if t == test):
        v = d[(test, n)]
        m = st.median(v)
        print(f"  {n:>3} {m / 1e3:9.1f}k {m / base:8.2f} {100 * m / base / n:4.0f}% {100 * (max(v) - min(v)) / m:6.1f}% "
              f"{min(v) / 1e3:9.1f}k {max(v) / 1e3:9.1f}k")
EOF
echo "results in $out"
