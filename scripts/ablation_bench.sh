#!/usr/bin/env bash
# Ablation: builds every variant in the table below, then runs them interleaved.
# usage: scripts/ablation_bench.sh [reps] [ms_per_run]
set -euo pipefail

cd "$(dirname "$0")/.."
reps=${1:-4}
ms=${2:-3000}
read -ra threads <<< "${THREADS:-1 8 16 32}"

order=(base compact single nocopy lookup all)
declare -A flags=(
    [base]=""
    [compact]="-DNANA_COMPACT_ACTIONS=1"
    [single]="-DNANA_SINGLE_PASS=1"
    [nocopy]="-DNANA_HISTORY_NOCOPY=1"
    [lookup]="-DNANA_ONE_LOOKUP=1"
    [all]="-DNANA_COMPACT_ACTIONS=1 -DNANA_SINGLE_PASS=1 -DNANA_HISTORY_NOCOPY=1 -DNANA_ONE_LOOKUP=1"
)
read -ra variants <<< "${VARIANTS:-${order[*]}}"
baseline=${variants[0]}

git submodule update --init --recursive
for v in "${variants[@]}"; do
    echo "building $v"
    cmake -S . -B "build-abl-$v" -DCMAKE_BUILD_TYPE=Release -DNANA_GUI=OFF \
        -DCMAKE_CXX_FLAGS="-DNANA_MPMCTS=1 ${flags[$v]}" > /dev/null
    cmake --build "build-abl-$v" --target nana_bench -j "$(nproc)" > /dev/null
done

out="results/$(hostname)-ablation-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out/raw"
{
    echo "host: $(hostname)  cpus: $(nproc)  threads: ${threads[*]}  reps: $reps  ms: $ms  variants: ${variants[*]}"
    lscpu 2>/dev/null | grep -E "Model name|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|L3" || true
    git log -1 --format="commit: %h %s"
    echo "uncommitted:"
    git status --short
    for v in "${variants[@]}"; do echo "$v: ${flags[$v]}"; done
} > "$out/machine.txt"

with_rss() {
    python3 -c 'import resource, subprocess, sys; subprocess.run(sys.argv[1:], check=True); print("max rss kib:", resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)' "$@"
}

nv=${#variants[@]}
for test in 0 1; do
    for n in "${threads[@]}"; do
        for ((r = 1; r <= reps; r++)); do
            for ((k = 0; k < nv; k++)); do
                v=${variants[$(( (k + r) % nv ))]}
                with_rss "build-abl-$v/cpp/nana_bench" "$n" "$ms" "$test" > "$out/raw/t$n-test$test-$v-rep$r.txt"
            done
            echo "test $test  threads $n  rep $r done"
        done
    done
done

python3 - "$out" "$baseline" "${variants[*]}" <<'EOF' | tee "$out/summary.txt"
import collections, csv, glob, os, re, statistics as st, sys
out, baseline, variants = sys.argv[1], sys.argv[2], sys.argv[3].split()
rows = []
for f in sorted(glob.glob(os.path.join(out, "raw", "*.txt"))):
    m = re.match(r"t(\d+)-test(\d)-(\w+)-rep(\d+)\.txt", os.path.basename(f))
    n, test, variant, rep = int(m[1]), m[2], m[3], int(m[4])
    text = open(f).read()
    num = lambda pat: float(re.search(pat, text, re.M)[1])
    prof = [l for l in text.splitlines() if re.match(r"^ +\d+ +[0-9.]+%", l)]
    def col(part, i):
        vals = [float(l.split("|")[part].split()[i].rstrip("%")) for l in prof]
        vals = [v for v in vals if v > 0]
        return st.mean(vals) if vals else 0.0
    mism = re.search(r"packed move mismatches (\d+)", text)
    rows.append(dict(threads=n, test=test, variant=variant, rep=rep,
                     sims_per_s=num(r"^simulations / second: ([0-9.e+]+)"),
                     wait_pct=col(0, 1), select_ns=col(3, 0), backprop_ns=col(3, 1), rollout_ns=col(3, 4),
                     queue_us=col(4, 0), depth=num(r"^tree depth: (\d+)"),
                     rss_mib=num(r"^max rss kib: (\d+)") / 1024,
                     move_mismatches=int(mism[1]) if mism else 0))
with open(os.path.join(out, "runs.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

d = collections.defaultdict(dict)
for r in rows:
    d[(r["test"], r["threads"], r["variant"])][r["rep"]] = r
med = lambda rs, k: st.median(r[k] for r in rs)
def ratio(test, n, v, ref, key="sims_per_s"):
    a, b = d.get((test, n, v), {}), d.get((test, n, ref), {})
    common = sorted(set(a) & set(b))
    return st.median(a[i][key] / b[i][key] for i in common) if common else float("nan")
print(f"median of reps; ratios are medians of per-rep ratios vs {baseline} (same rep = run back to back)")
print("rss/sim: memory per simulation run, vs baseline (the tree grows with simulations)")
for test in sorted({k[0] for k in d}):
    print(f"\ntest {test}")
    print(f"  {'thr':>3} {'variant':<8} {'sims/s':>9} {'ratio':>6} {'wait':>6} {'select':>7} {'backprop':>8} {'rollout':>8} {'queue us':>8} {'rss MiB':>8} {'rss/sim':>7} {'depth':>5} {'bad moves':>9}")
    for n in sorted({k[1] for k in d if k[0] == test}):
        for v in variants:
            rs = list(d.get((test, n, v), {}).values())
            if not rs:
                continue
            r_sims = ratio(test, n, v, baseline) if v != baseline else float("nan")
            a, b = d.get((test, n, v), {}), d.get((test, n, baseline), {})
            common = sorted(set(a) & set(b))
            per_sim = (st.median((a[i]["rss_mib"] / a[i]["sims_per_s"]) / (b[i]["rss_mib"] / b[i]["sims_per_s"]) for i in common)
                       if common and v != baseline else float("nan"))
            fmt = lambda x: "" if x != x else f"{x:.3f}"
            print(f"  {n:>3} {v:<8} {med(rs, 'sims_per_s') / 1e3:8.1f}k {fmt(r_sims):>6} {med(rs, 'wait_pct'):5.1f}% "
                  f"{med(rs, 'select_ns'):7.0f} {med(rs, 'backprop_ns'):8.0f} {med(rs, 'rollout_ns'):8.0f} {med(rs, 'queue_us'):8.1f} "
                  f"{med(rs, 'rss_mib'):8.0f} {fmt(per_sim):>7} {med(rs, 'depth'):5.0f} {sum(r['move_mismatches'] for r in rs):9d}")
EOF
echo "results in $out"
