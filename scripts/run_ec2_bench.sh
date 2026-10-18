#!/usr/bin/env bash
# Run scripts/bench_scaling.sh on an EC2 instance, copy the results back, then shut it down.
#
# usage: scripts/run_ec2_bench.sh <host> <key.pem> [ref] [bench args...]
#   host   public DNS or IP of a running Ubuntu 24.04 instance (user: $EC2_USER, default ubuntu)
#   key    the instance's ssh key
#   ref    branch or commit to benchmark; must be pushed to origin (default: current branch)
#   bench args are passed to bench_scaling.sh: [max_threads] [reps] [ms_per_run]
#
# env:
#   EC2_USER      ssh user (default ubuntu)
#   MAX_MINUTES   the instance shuts itself down after this long no matter what (default 180),
#                 so a crashed sweep or a sleeping laptop can't leave it running
#   POLL_SECONDS  how often to check whether the sweep is done (default 60)
#
# The instance is shut down from inside, which stops it; launch it with
# "shutdown behavior: terminate" if you want it terminated instead.
# Results land in ec2-results/<timestamp>/.
set -uo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,20p' "$0"
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

ssh_opts=(-i "$key" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=20)
remote() { ssh "${ssh_opts[@]}" "$user@$host" "$@"; }

local_dir="ec2-results/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$local_dir"

shutdown_instance() {
    echo "shutting the instance down"
    remote "sudo shutdown -h now" || echo "could not reach the instance to shut it down; it will stop itself after $max_minutes minutes"
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
remote bash -s <<EOF || { echo "setup failed"; shutdown_instance; exit 1; }
set -euo pipefail
# dead man's switch: stop no matter what after max_minutes
sudo shutdown -h +$max_minutes "nana bench time limit" || true
sudo apt-get update -q
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q git cmake g++ python3 > /dev/null
rm -rf nana
git clone -q "$repo" nana
cd nana
git checkout -q "$commit"
nohup bash -c 'bash scripts/bench_scaling.sh $bench_args; echo \$? > ../bench.done' > ../bench.log 2>&1 < /dev/null &
EOF

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
scp "${ssh_opts[@]}" -r "$user@$host:nana/results/*" "$local_dir/" || echo "could not copy results"

shutdown_instance

for summary in "$local_dir"/*/summary.txt; do
    [ -f "$summary" ] && cat "$summary"
done
echo "results in $local_dir"
