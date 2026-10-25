#!/usr/bin/env bash
# Run a benchmark script on an EC2 instance, copy the results back, then shut it down.
# usage: scripts/run_ec2_bench.sh <host> <key.pem> [ref] [bench args...]
set -uo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,3p' "$0"
    exit 1
fi

host=$1
key=$2
ref=${3:-$(git rev-parse --abbrev-ref HEAD)}
shift $(( $# < 3 ? $# : 3 ))
bench_args="$*"
user=${EC2_USER:-ubuntu}
max_minutes=${MAX_MINUTES:-180}
poll=${POLL_SECONDS:-60}
repo=$(git remote get-url origin)
script=${SCRIPT:-bench_scaling.sh}
worktree=${WORKTREE:-0}
self_shutdown=${SELF_SHUTDOWN:-0}

ssh_opts=(-i "$key" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=20)
remote() { ssh "${ssh_opts[@]}" "$user@$host" "$@"; }

local_dir="ec2-results/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$local_dir"

armed=0  # set once the instance's own shutdown timer is scheduled
shutdown_instance() {
    echo "shutting the instance down"
    if ! remote "sudo shutdown -h now"; then
        if [ "$armed" = 1 ]; then
            echo "could not reach the instance to shut it down; it will stop itself after $max_minutes minutes"
        else
            echo "could not reach the instance and its shutdown timer was never set: STOP IT MANUALLY"
        fi
    fi
}

if ! git ls-remote --exit-code origin "$ref" > /dev/null 2>&1 && ! git cat-file -e "$ref^{commit}" 2>/dev/null; then
    echo "ref $ref not found; push it to origin first"
    exit 1
fi
commit=$(git rev-parse "$ref")
if ! git branch -r --contains "$commit" 2>/dev/null | grep -q .; then
    echo "warning: $commit doesn't look pushed to origin; the instance clones from $repo"
fi

echo "setting up $host (benchmarking $ref = $commit)"
scp "${ssh_opts[@]}" "$(dirname "$0")/$script" "$user@$host:$script" || { echo "upload failed"; shutdown_instance; exit 1; }
extract=":"
if [ "$worktree" = 1 ]; then
    tarball=ec2-results/worktree-upload.tgz  # gitignored, and a relative path keeps tar and scp happy on Windows
    upload=$( { git ls-files -m -o --exclude-standard
                git submodule foreach --quiet --recursive 'git ls-files -m | sed "s|^|$displaypath/|"'
              } | while read -r f; do [ -f "$f" ] && echo "$f"; done)
    tar -czf "$tarball" -T - <<< "$upload"
    echo "uploading uncommitted files: $(tr '\n' ' ' <<< "$upload")"
    scp "${ssh_opts[@]}" "$tarball" "$user@$host:worktree.tgz" || { echo "upload failed"; shutdown_instance; exit 1; }
    rm -f "$tarball"
    extract="tar -xzf ../worktree.tgz && sed -i 's/\r\$//' scripts/*.sh"
fi
remote bash -s <<EOF || { echo "setup failed"; shutdown_instance; exit 1; }
set -euo pipefail
sudo shutdown -h +$max_minutes "nana bench time limit" || true
sudo apt-get update -q
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q git cmake g++ python3 > /dev/null
if [ -d nana/results ] && [ -n "\$(ls -A nana/results 2> /dev/null)" ]; then
    mkdir -p results-archive
    mv nana/results "results-archive/\$(date +%Y%m%d-%H%M%S)"
fi
rm -rf nana
git clone -q "$repo" nana
cd nana
git checkout -q "$commit"
git submodule update -q --init --recursive
$extract
sed 's/\r\$//' ../$script > scripts/$script
rm -f ../bench.done
nohup bash -c 'THREADS="${THREADS:-}" bash scripts/$script $bench_args; status=\$?; echo \$status > ../bench.done; [ "$self_shutdown" = 1 ] && sudo shutdown -h +\$(( status == 0 ? 10 : 30 )) "nana bench done"' > ../bench.log 2>&1 < /dev/null &
EOF

armed=1
echo "sweep running; polling every ${poll}s (instance stops itself after $max_minutes minutes regardless)"
while true; do
    sleep "$poll"
    if status=$(remote "cat bench.done 2>/dev/null || true"); then
        if [ -n "$status" ]; then
            echo "sweep finished with exit code $status"
            break
        fi
        remote "tail -n 1 bench.log" 2>/dev/null || true
    else
        echo "instance unreachable, retrying"
    fi
done

echo "copying results to $local_dir"
scp "${ssh_opts[@]}" "$user@$host:bench.log" "$local_dir/" || echo "could not copy bench.log"
if remote "cd nana/results && tar -czf - ." > "$local_dir/results.tgz"; then
    tar -xzf "$local_dir/results.tgz" -C "$local_dir" && rm "$local_dir/results.tgz"
else
    echo "could not copy results"
fi

shutdown_instance

for summary in "$local_dir"/*/summary.txt; do
    [ -f "$summary" ] && cat "$summary"
done
echo "results in $local_dir"
