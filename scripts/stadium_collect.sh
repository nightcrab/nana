#!/usr/bin/env bash
# Collect games with Universal-Shaktris-Stadium for fitting opponent models. nana plays P1; P2 is
# nana too, or the TBP bot at $BOT2 (e.g. Cold Clear's cc-tbp built with tetrio_garbage).
# usage: scripts/stadium_collect.sh [minutes] [pps] [threads per bot] [thinks per move]
#        (defaults 120, 1, 1, 0; 0 thinks leaves each bot its own budget)
set -euo pipefail

minutes=${1:-120}
pps=${2:-1}
threads=${3:-1}
thinks=${4:-0}
bot2=${BOT2:-}
if [ -n "$bot2" ]; then
    bot2=$(realpath "$bot2")
    [ -x "$bot2" ] || { echo "BOT2 $bot2 isn't executable" >&2; exit 1; }
fi
cpus=$(nproc)
stadiums=${STADIUMS:-$(( cpus / (2 * threads) ))}
root=$(pwd)
out="$root/results/stadium-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out"

echo "building nana's tbp bot"
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNANA_GUI=OFF > /dev/null
cmake --build build --target nana_tbp -j "$cpus" > /dev/null
bot="$root/build/cpp/nana_tbp"
python3 - "$root/build/cpp/nana.json" "$threads" <<'EOF'
import json, os, sys
path = sys.argv[1]
settings = json.load(open(path)) if os.path.exists(path) else {}
settings.pop("markov_opponent", None)
settings.update(threads=int(sys.argv[2]), rules="tetrio")
with open(path, "w") as file:
    json.dump(settings, file)
EOF

echo "building stadium_cli"
rm -rf stadium
git clone -q --depth 1 https://github.com/nightcrab/Universal-Shaktris-Stadium stadium
(cd stadium && git log -1 --format="stadium commit %h %s")
(cd stadium/src && gcc -O2 -c sqlite3/sqlite3.c -o sqlite3.o \
    && g++ -O2 -std=c++23 -include variant -I. -IShaktris -ITBP -IUtil -Isqlite3 \
        stadium_cli.cpp TBP/Bot.cpp Shaktris/Game.cpp sqlite3.o -o stadium_cli -lpthread -ldl)
stadium="$root/stadium/src/stadium_cli"

cat > "$out/config.txt" <<EOF
minutes $minutes
pps $pps
threads per bot $threads
thinks per move $thinks
p2 ${bot2:-nana}
stadiums $stadiums
rules tetrio (the stadium plays TETR.IO damage whatever -game_type says)
nana commit $(git rev-parse HEAD) (plus uploaded worktree changes)
EOF
cat "$out/config.txt"

end=$(( $(date +%s) + minutes * 60 ))

run_stadium() {
    local k=$1
    local dir="$out/stadium_$k"
    mkdir -p "$dir"
    local first=$(( k * 2 * threads ))
    for side in 1 2; do
        local lo=$(( first + (side - 1) * threads ))
        local exe=$bot
        [ "$side" = 2 ] && [ -n "$bot2" ] && exe=$bot2
        cat > "$dir/bot$side.sh" <<EOF
#!/bin/sh
exec taskset -c $lo-$(( lo + threads - 1 )) "$exe"
EOF
        chmod +x "$dir/bot$side.sh"
    done
    cat > "$dir/stadium_cli_settings.json" <<EOF
{"num_games": 1000000, "db_path": "$out/stadium_$k.db", "min_nodes": 2000, "pps": $pps, "threads": $threads, "thinks": $thinks}
EOF
    local restarts=0
    while [ "$(date +%s)" -lt "$end" ]; do
        local left=$(( end - $(date +%s) ))
        (cd "$dir" && timeout -s INT "$left" "$stadium" "$dir/bot1.sh" "$dir/bot2.sh" > /dev/null 2>> "$dir/stderr.log") || true
        [ "$(date +%s)" -lt "$end" ] && restarts=$(( restarts + 1 ))
        sleep 1
    done
    echo "$restarts" > "$dir/restarts"
}

echo "running $stadiums stadiums for $minutes minutes"
for (( k = 0; k < stadiums; k++ )); do
    run_stadium "$k" &
done
wait
pkill -f "$bot" || true
[ -z "$bot2" ] || pkill -f "$bot2" || true

python3 - "$out" <<'EOF' | tee "$out/summary.txt"
import glob, os, sqlite3, sys
out = sys.argv[1]
games = moves = 0
for path in sorted(glob.glob(os.path.join(out, "stadium_*.db"))):
    db = sqlite3.connect(path)
    try:
        g, m = db.execute("SELECT COUNT(DISTINCT game_id), COUNT(*) FROM Data WHERE state = 'PLAYING'").fetchone()
    except sqlite3.Error:
        g = m = 0
    games += g
    moves += m
restarts = sum(int(open(p).read() or 0) for p in glob.glob(os.path.join(out, "stadium_*", "restarts")))
print(f"{games} games, {moves} turns ({2 * moves} player-steps), {restarts} stadium restarts")
EOF
