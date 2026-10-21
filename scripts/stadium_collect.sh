#!/usr/bin/env bash
# Collect nana-vs-nana games for fitting MarkovOpponent: runs Universal-Shaktris-Stadium's stadium_cli
# in parallel, each bot pinned to its own cpu, until the time budget runs out. Each stadium writes a
# SQLite database that scripts/train_markov_opponent.py reads directly.
#
# usage: scripts/stadium_collect.sh [minutes] [pps] [threads per bot]   (defaults 120, 1, 1)
#
# env:
#   STADIUMS   stadiums to run at once (default: cpus / (2 * threads per bot))
#   MARKOV     NANA_MARKOV_OPPONENT for the bots: 1 = search with the HMM opponent (default 1)
#
# Run it on EC2 with scripts/run_ec2_bench.sh:
#   WORKTREE=1 SCRIPT=stadium_collect.sh MAX_MINUTES=150 scripts/run_ec2_bench.sh <host> <key> master 120 1 1
# Results land in results/stadium-<timestamp>/: one stadium_<k>.db per stadium and summary.txt.
set -euo pipefail

minutes=${1:-120}
pps=${2:-1}
threads=${3:-1}
markov=${MARKOV:-1}
cpus=$(nproc)
stadiums=${STADIUMS:-$(( cpus / (2 * threads) ))}
root=$(pwd)
out="$root/results/stadium-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out"

echo "building nana's tbp bot"
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNANA_GUI=OFF > /dev/null
cmake --build build --target tbp_tetris -j "$cpus" > /dev/null
bot="$root/build/cpp/tbp_tetris"

echo "building stadium_cli"
rm -rf stadium
git clone -q --depth 1 https://github.com/shakkar23/Universal-Shaktris-Stadium stadium
(cd stadium && git log -1 --format="stadium commit %h %s")
# upstream skips the game's final move when logging, but its attacks are what kill the loser
python3 - <<'EOF'
p = "stadium/src/stadium_cli.cpp"
s = open(p, newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"
anchor = "\t\t\t\tif(game.game_over) {" + nl + "\t\t\t\t\tgame_state = State::GAME_OVER;"
assert anchor in s, "stadium_cli.cpp changed; update the final-move patch"
patch = nl.join([
    "\t\t\t\tif(game.game_over) {",
    "\t\t\t\t\t// record the final move like any other: its attacks are what killed the loser",
    "\t\t\t\t\tp1.attack = game.p1_damage_sent;",
    "\t\t\t\t\tp1.damage_received = game.p2_damage_sent;",
    "\t\t\t\t\tp1.spun = game.p1_spun;",
    "\t\t\t\t\tp2.attack = game.p2_damage_sent;",
    "\t\t\t\t\tp2.damage_received = game.p1_damage_sent;",
    "\t\t\t\t\tp2.spun = game.p2_spun;",
    "\t\t\t\t\tgame_states.push_back({ s, p1, p2, game_uuid, move_index });",
    "\t\t\t\t\tmove_index++;",
    "\t\t\t\t\tgame_state = State::GAME_OVER;",
])
open(p, "w", newline="").write(s.replace(anchor, patch, 1))
EOF
# its CMakeLists needs SDL2 for every target, so build the cli directly
(cd stadium/src && gcc -O2 -c sqlite3/sqlite3.c -o sqlite3.o \
    && g++ -O2 -std=c++23 -include variant -I. -IShaktris -ITBP -IUtil -Isqlite3 \
        stadium_cli.cpp TBP/Bot.cpp Shaktris/Game.cpp sqlite3.o -o stadium_cli -lpthread -ldl)
stadium="$root/stadium/src/stadium_cli"

cat > "$out/config.txt" <<EOF
minutes $minutes
pps $pps
threads per bot $threads
stadiums $stadiums
NANA_MARKOV_OPPONENT $markov
NANA_RULES tetrio (the stadium plays TETR.IO damage whatever -game_type says)
nana commit $(git rev-parse HEAD) (plus uploaded worktree changes)
EOF
cat "$out/config.txt"

end=$(( $(date +%s) + minutes * 60 ))

# one stadium: two bots on their own cpus, restarted if the stadium dies (it throws on an invalid
# placement) until the time is up. games are written to the database as they finish
run_stadium() {
    local k=$1
    local dir="$out/stadium_$k"
    mkdir -p "$dir"
    local first=$(( k * 2 * threads ))
    for side in 1 2; do
        local lo=$(( first + (side - 1) * threads ))
        cat > "$dir/bot$side.sh" <<EOF
#!/bin/sh
export NANA_THREADS=$threads NANA_MARKOV_OPPONENT=$markov NANA_RULES=tetrio
exec taskset -c $lo-$(( lo + threads - 1 )) "$bot"
EOF
        chmod +x "$dir/bot$side.sh"
    done
    cat > "$dir/stadium_cli_settings.json" <<EOF
{"num_games": 1000000, "db_path": "$out/stadium_$k.db", "min_nodes": 2000, "pps": $pps}
EOF
    local restarts=0
    while [ "$(date +%s)" -lt "$end" ]; do
        local left=$(( end - $(date +%s) ))
        # SIGINT: the stadium closes the database; a game in progress is lost
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
# bots whose stadium was interrupted exit on end of input; make sure
pkill -f "$bot" || true

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
