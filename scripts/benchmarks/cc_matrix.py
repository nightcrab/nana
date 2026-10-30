"""nana vs Cold Clear win-rate matrix on stadium, fixed time per piece.

usage: python3 scripts/benchmarks/cc_matrix.py OUT NANA_TBP CC_TBP STADIUM [--pps 20] [--games 256] [--threads 1,2,4,8,16]
       python3 scripts/benchmarks/cc_matrix.py OUT --summary

Each cell (nana threads, cc threads) is two stadium jobs of games/2, one with nana as player 1 and
one with Cold Clear as player 1. Jobs are packed onto the 32 cores: a bot with up to 8 threads sits
inside one 8-core chiplet, a 16-thread bot takes two whole chiplets. Each stadium run writes its
own runN.out; a crashed stadium is restarted for the games it still owes, and the win tally is
the sum of the last tally line of every run. Rerunning into the same OUT adds games to every cell
(jobs are numbered by batch), so cells can be aggregated later.
"""
import argparse, glob, json, os, re, signal, subprocess, sys, time

CHIPLETS = [list(range(8 * k, 8 * k + 8)) for k in range(4)]
TALLY = re.compile(r"Player 1 wins: (\d+)\s+Player 2 wins: (\d+)\s+Draws: (\d+)\s+Total games: (\d+)")


def tally(job_dir):
    w1 = w2 = dr = 0
    for path in glob.glob(os.path.join(job_dir, "run*.out")):
        found = TALLY.findall(open(path, errors="replace").read())
        if found:
            a, b, c, _ = map(int, found[-1])
            w1 += a; w2 += b; dr += c
    return w1, w2, dr


class Cores:
    def __init__(self):
        self.free = [set(c) for c in CHIPLETS]

    def take(self, t):
        if t > 8:
            whole = [k for k, f in enumerate(self.free) if len(f) == 8]
            if len(whole) * 8 < t:
                return None
            out = []
            for k in whole[: t // 8]:
                out += sorted(self.free[k]); self.free[k] = set()
            return out
        fits = [k for k, f in enumerate(self.free) if len(f) >= t]
        if not fits:
            return None
        k = min(fits, key=lambda k: len(self.free[k]))  # best fit
        out = sorted(self.free[k])[:t]
        self.free[k] -= set(out)
        return out

    def give(self, cores):
        for c in cores:
            self.free[c // 8].add(c)


def job_dirs(out):
    return sorted(glob.glob(os.path.join(out, "cells", "n*_c*", "b*_p1*")))


def summary(out):
    T = json.load(open(os.path.join(out, "matrix.json")))["threads"]
    cells = {}
    for d in job_dirs(out):
        cell = os.path.basename(os.path.dirname(d))
        p1 = json.load(open(os.path.join(d, "job.json")))["p1"]
        w1, w2, dr = tally(d)
        nana, cc = (w1, w2) if p1 == "nana" else (w2, w1)
        c = cells.setdefault(cell, [0, 0, 0])
        c[0] += nana; c[1] += cc; c[2] += dr
    lines = ["nana score vs Cold Clear (wins + draws/2, over games played), 95% Wilson interval below",
             "rows: nana threads; columns: Cold Clear threads", ""]
    head = "nana\\cc " + "".join(f"{c:>16}" for c in T)
    lines.append(head)
    for n in T:
        row, ci = f"{n:>7} ", " " * 8
        for c in T:
            nana, cc, dr = cells.get(f"n{n}_c{c}", [0, 0, 0])
            g = nana + cc + dr
            if g:
                p = (nana + dr / 2) / g
                z = 1.96; den = 1 + z * z / g
                mid = (p + z * z / (2 * g)) / den
                half = z * ((p * (1 - p) / g + z * z / (4 * g * g)) ** 0.5) / den
                row += f"{100 * p:>10.1f}% {g:>4}"
                ci += f"{100 * (mid - half):>9.1f}-{100 * (mid + half):.1f}"
            else:
                row += f"{'-':>16}"; ci += " " * 16
        lines += [row, ci]
    lines += ["", "cell: score% games; draws count half"]
    text = "\n".join(lines)
    open(os.path.join(out, "summary.txt"), "w").write(text + "\n")
    json.dump(cells, open(os.path.join(out, "cells.json"), "w"), indent=1)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("nana", nargs="?")
    ap.add_argument("cc", nargs="?")
    ap.add_argument("stadium", nargs="?")
    ap.add_argument("--pps", type=float, default=20)
    ap.add_argument("--games", type=int, default=256)
    ap.add_argument("--threads", default="1,2,4,8,16")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    if a.summary:
        print(summary(a.out)); return
    T = [int(t) for t in a.threads.split(",")]
    os.makedirs(os.path.join(a.out, "cells"), exist_ok=True)
    meta_path = os.path.join(a.out, "matrix.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"threads": T, "pps": a.pps, "batches": []}
    assert meta["threads"] == T and meta["pps"] == a.pps, "a rerun must keep threads and pps"
    batch = len(meta["batches"])
    meta["batches"].append({"games_per_cell": a.games, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "nana": a.nana, "cc": a.cc, "stadium": a.stadium})
    json.dump(meta, open(meta_path, "w"), indent=1)
    bots = {"nana": os.path.abspath(a.nana), "cc": os.path.abspath(a.cc)}
    jobs = []
    for n in T:
        for c in T:
            for p1 in ("nana", "cc"):
                d = os.path.join(a.out, "cells", f"n{n}_c{c}", f"b{batch}_p1{p1}")
                os.makedirs(d, exist_ok=True)
                t1, t2 = (n, c) if p1 == "nana" else (c, n)
                job = {"dir": d, "p1": p1, "threads": [t1, t2], "games": a.games // 2, "runs": 0}
                json.dump({k: job[k] for k in ("p1", "threads", "games")}, open(os.path.join(d, "job.json"), "w"))
                jobs.append(job)
    jobs.sort(key=lambda j: -sum(j["threads"]))
    cores, running, pending = Cores(), [], list(jobs)
    t0 = time.time()

    def launch(job, alloc):
        d = job["dir"]
        left = job["games"] - sum(tally(d)[:3])
        a1, a2 = alloc
        for side, (name, cs) in enumerate(((job["p1"], a1), ("cc" if job["p1"] == "nana" else "nana", a2)), 1):
            with open(os.path.join(d, f"bot{side}.sh"), "w") as f:
                f.write(f"#!/bin/sh\nexec taskset -c {','.join(map(str, cs))} \"{bots[name]}\"\n")
            os.chmod(os.path.join(d, f"bot{side}.sh"), 0o755)
        r = job["runs"]
        json.dump({"num_games": left, "db_path": os.path.join(d, f"games{r}.db"), "min_nodes": 0,
                   "pps": a.pps, "threads": job["threads"]}, open(os.path.join(d, "stadium_cli_settings.json"), "w"))
        out = open(os.path.join(d, f"run{r}.out"), "w"); err = open(os.path.join(d, "stderr.log"), "a")
        job["proc"] = subprocess.Popen([os.path.abspath(a.stadium), os.path.join(d, "bot1.sh"), os.path.join(d, "bot2.sh")],
                                       cwd=d, stdout=out, stderr=err, start_new_session=True)
        job["alloc"] = alloc; job["runs"] = r + 1
        open(os.path.join(d, "cores.txt"), "a").write(f"run{r}: p1 {a1} p2 {a2}\n")

    while pending or running:
        for job in list(running):
            if job["proc"].poll() is None:
                continue
            running.remove(job)
            cores.give(job["alloc"][0] + job["alloc"][1])
            try:
                os.killpg(job["proc"].pid, signal.SIGKILL)  # bots left behind by a crashed stadium
            except ProcessLookupError:
                pass
            if sum(tally(job["dir"])) < job["games"]:
                if job["runs"] >= 20:
                    print(f"giving up on {job['dir']} after 20 stadium runs", flush=True)
                else:
                    pending.insert(0, job)
        for job in list(pending):
            a1 = cores.take(job["threads"][0])
            if a1 is None:
                continue
            a2 = cores.take(job["threads"][1])
            if a2 is None:
                cores.give(a1); continue
            pending.remove(job); running.append(job)
            launch(job, (a1, a2))
        done = sum(1 for j in jobs if sum(tally(j["dir"])) >= j["games"])
        status = (f"{time.strftime('%H:%M:%S')} {(time.time() - t0) / 60:.1f} min: {done}/{len(jobs)} jobs done, "
                  f"{len(running)} running, {len(pending)} waiting, games {sum(sum(tally(j['dir'])) for j in jobs)}"
                  f"/{sum(j['games'] for j in jobs)}")
        open(os.path.join(a.out, "status.txt"), "w").write(status + "\n")
        time.sleep(5)
    print(status, flush=True)
    print(summary(a.out), flush=True)


if __name__ == "__main__":
    main()
