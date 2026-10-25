#!/usr/bin/env bash
# What does crossing chiplets cost? Packed against spread pinning.
# usage: scripts/chiplet_bench.sh [reps] [ms_per_run]
set -euo pipefail

cd "$(dirname "$0")/.."
reps=${1:-5}
ms=${2:-3000}
read -ra threads <<< "${THREADS:-4 8 16}"

git submodule update --init --recursive
cmake -S . -B build-chiplet -DCMAKE_BUILD_TYPE=Release -DNANA_GUI=OFF -DCMAKE_CXX_FLAGS="-DNANA_MPMCTS=1" > /dev/null
cmake --build build-chiplet --target nana_bench -j "$(nproc)" > /dev/null
bench=$PWD/build-chiplet/cpp/nana_bench

out="results/$(hostname)-chiplet-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out/raw"
{
    echo "host: $(hostname)  cpus: $(nproc)  threads: ${threads[*]}  reps: $reps  ms: $ms"
    lscpu 2>/dev/null | grep -E "Model name|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|L3" || true
    for c in 0 8 16 24; do
        echo "cpu$c L3 shared with: $(cat /sys/devices/system/cpu/cpu$c/cache/index3/shared_cpu_list 2>/dev/null)"
    done
    git log -1 --format="commit: %h %s"
    echo "uncommitted:"
    git status --short
} > "$out/machine.txt"

for test in 0 1; do
    for n in "${threads[@]}"; do
        for ((r = 1; r <= reps; r++)); do
            if (( r % 2 )); then layouts="packed spread"; else layouts="spread packed"; fi
            for layout in $layouts; do
                f="$out/raw/t$n-test$test-$layout-rep$r.txt"
                if [ "$layout" = spread ]; then
                    NANA_PIN_LAYOUT=spread "$bench" "$n" "$ms" "$test" > "$f"
                else
                    "$bench" "$n" "$ms" "$test" > "$f"
                fi
            done
            echo "test $test  threads $n  rep $r done"
        done
    done
done

python3 - "$out" <<'EOF' | tee "$out/summary.txt"
import collections, csv, glob, os, re, statistics as st, sys
out = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(out, "raw", "*.txt"))):
    m = re.match(r"t(\d+)-test(\d)-(\w+)-rep(\d+)\.txt", os.path.basename(f))
    n, test, layout, rep = int(m[1]), m[2], m[3], int(m[4])
    text = open(f).read()
    get = lambda key: float(re.search(r"^" + re.escape(key) + r": ([0-9.e+-]+)", text, re.M)[1])
    prof = [l for l in text.splitlines() if re.match(r"^ +\d+ +[0-9.]+%", l)]
    def col(part, i):
        vals = [float(l.split("|")[part].split()[i].rstrip("%")) for l in prof]
        vals = [v for v in vals if v > 0]
        return st.mean(vals) if vals else 0.0
    rows.append(dict(threads=n, test=test, layout=layout, rep=rep, sims_per_s=get("simulations / second"),
                     l3_groups=int(get("L3 cache groups across workers")), wait_pct=col(0, 1),
                     select_ns=col(3, 0), backprop_ns=col(3, 1), rollout_ns=col(3, 4), queue_us=col(4, 0),
                     stolen_pct=float(re.search(r"^stolen rollouts: ([0-9.]+)%", text, re.M)[1])))
with open(os.path.join(out, "runs.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

d = collections.defaultdict(list)
for r in rows:
    d[(r["test"], r["threads"], r["layout"])].append(r)
med = lambda rs, k: st.median(r[k] for r in rs)
print("median of reps; ratio = median of per-rep spread/packed; ns = mean over threads, per job")
for test in sorted({k[0] for k in d}):
    print(f"\ntest {test}")
    print(f"  {'thr':>3} {'layout':<7} {'L3':>3} {'sims/s':>9} {'ratio':>6} {'wait':>6} {'select ns':>9} {'backprop ns':>11} {'rollout ns':>10} {'queue us':>8} {'stolen':>7}")
    for n in sorted({k[1] for k in d if k[0] == test}):
        p, s = d[(test, n, "packed")], d[(test, n, "spread")]
        pair = {(r["rep"]): r for r in p}
        ratio = st.median(r["sims_per_s"] / pair[r["rep"]]["sims_per_s"] for r in s if r["rep"] in pair)
        for name, rs in (("packed", p), ("spread", s)):
            print(f"  {n:>3} {name:<7} {med(rs, 'l3_groups'):>3.0f} {med(rs, 'sims_per_s') / 1e3:8.1f}k "
                  f"{(f'{ratio:.3f}' if name == 'spread' else ''):>6} {med(rs, 'wait_pct'):5.1f}% {med(rs, 'select_ns'):9.0f} "
                  f"{med(rs, 'backprop_ns'):11.0f} {med(rs, 'rollout_ns'):10.0f} {med(rs, 'queue_us'):8.1f} {med(rs, 'stolen_pct'):6.1f}%")
EOF
echo "results in $out"
