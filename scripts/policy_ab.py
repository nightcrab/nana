"""Paired rank-vs-trained-softmax emulator A/B at fixed sims and fixed time."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time


def summary(rows, horizon):
    result = {"pairs": len(rows)}
    if not rows:
        return result
    for arm in ("rank", "softmax"):
        games = [r[arm] for r in rows]
        pieces = sum(g["pieces"] for g in games)
        think = sum(g["pieces"] * g["ms_per_move"] for g in games)
        result[arm] = {
            "attack_per_horizon_piece": statistics.mean(g["attack"] / horizon for g in games),
            "survival_rate": statistics.mean(not g["died"] for g in games),
            "mean_pieces": statistics.mean(g["pieces"] for g in games),
            "ms_per_move": think / max(pieces, 1),
            "simulations_per_second": sum(g["pieces"] * g["sims_per_move"] for g in games) * 1000 / max(think, 1e-9),
        }
    differences = [(r["softmax"]["attack"] - r["rank"]["attack"]) / horizon for r in rows]
    mean = statistics.mean(differences)
    se = statistics.stdev(differences) / math.sqrt(len(rows)) if len(rows) > 1 else None
    result["paired_attack_difference"] = mean
    result["paired_95pct_normal_CI"] = [mean - 1.96 * se, mean + 1.96 * se] if se is not None else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nana", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cpus", default="0,2,4,6,8,10,12,14")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--wall-games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1000000)
    parser.add_argument("--sims", type=int, default=1000)
    parser.add_argument("--ms", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=100)
    options = parser.parse_args()
    cpus = list(map(int, options.cpus.split(",")))
    if (not cpus or len(set(cpus)) != len(cpus) or min(cpus) < 0 or
            min(options.games, options.sims, options.ms, options.horizon) < 1 or options.wall_games < 0):
        parser.error("positive budgets/games/horizon and unique nonnegative CPUs are required")
    options.out.mkdir(parents=True, exist_ok=False)
    executable, model = options.nana.resolve(), options.model.resolve()
    baseline = json.loads(options.config.read_text())
    baseline.update(policy="rank", policy_model="")
    candidate = dict(baseline, policy="softmax", policy_model=str(model))
    configs = {}
    for arm, config in (("rank", baseline), ("softmax", candidate)):
        configs[arm] = (options.out / f"{arm}.json").resolve()
        configs[arm].write_text(json.dumps(config, indent=2))
        check = subprocess.run([str(executable), "--print-config"],
                               env=dict(os.environ, NANA_CONFIG=str(configs[arm])),
                               capture_output=True, text=True, check=True, timeout=60)
        loaded = json.loads(check.stdout)
        assert loaded["policy"] == config["policy"]
        (options.out / f"{arm}-loaded.json").write_text(json.dumps(loaded, indent=2))
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(options).items()}
    manifest.update(binary_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                    model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                    config_sha256=hashlib.sha256(options.config.read_bytes()).hexdigest(),
                    fitness="attack / fixed horizon; death forfeits remaining pieces",
                    intervals="paired normal approximation; descriptive, no early stopping")
    (options.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    started = time.time()
    rows = {"sims": [], "time": []}
    lock = threading.Lock()

    def save(phase):
        report = {"phase": phase, "elapsed_seconds": time.time() - started,
                  "sims": summary(rows["sims"], options.horizon),
                  "time": summary(rows["time"], options.horizon)}
        path = options.out / "status.tmp"
        path.write_text(json.dumps(report, indent=2))
        path.replace(options.out / "status.json")

    with (options.out / "pairs.jsonl").open("a", buffering=1) as log:
        for stage, games, seed0 in (("sims", options.games, options.seed),
                                    ("time", options.wall_games, options.seed + options.games)):
            def worker(cpu, index):
                for offset in range(index, games, len(cpus)):
                    seed = seed0 + offset
                    pair = {"seed": seed, "stage": stage, "cpu": cpu}
                    order = (offset + offset // len(cpus)) % 2
                    for arm in (("rank", "softmax") if order == 0 else ("softmax", "rank")):
                        command = [str(executable), "--threads", "1", "--games", "1", "--seed", str(seed),
                                   "--search-seed", "1", "--cpu", str(cpu), "--max-pieces", str(options.horizon)]
                        command += ["--sims", str(options.sims)] if stage == "sims" else ["--ms", str(options.ms)]
                        process = subprocess.run(command, env=dict(os.environ, NANA_CONFIG=str(configs[arm])),
                                                 capture_output=True, text=True, timeout=600)
                        if process.returncode:
                            raise RuntimeError(f"{stage} seed {seed} {arm}: {process.stderr}")
                        pair[arm] = json.loads(process.stdout)
                    with lock:
                        rows[stage].append(pair)
                        log.write(json.dumps(pair) + "\n")
                        save("running")
                        if len(rows[stage]) % 25 == 0:
                            print(json.dumps({"stage": stage, "pairs": len(rows[stage])}), flush=True)

            save("running")
            with ThreadPoolExecutor(max_workers=len(cpus)) as pool:
                jobs = [pool.submit(worker, cpu, i) for i, cpu in enumerate(cpus)]
                for job in jobs:
                    job.result()
        save("complete")
    print((options.out / "status.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
