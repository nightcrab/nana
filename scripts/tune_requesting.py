#!/usr/bin/env python3
"""Random search over the work-requesting and stealing heuristics, scored by simulations/s.

usage: python3 scripts/tune_requesting.py --bench build-bench-mp1/cpp/nana_bench --threads 32
"""
import argparse
import csv
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time

DEFAULTS = {"message_weight": 1, "rollout_weight": 30, "low_water": 30, "give_margin": 3,
            "backoff_iterations": 64, "steal_chiplet_penalty": 30, "steal_topk": 4, "steal_temperature": 10}


def sample(rng):
    rollout_weight = round(math.exp(rng.uniform(math.log(5), math.log(120))))
    temperature = 0 if rng.random() < 0.2 else math.exp(rng.uniform(math.log(0.05), math.log(2)))
    return {
        "message_weight": rng.choice((0, 1)),
        "rollout_weight": rollout_weight,
        "low_water": round(rng.uniform(0, 4) * rollout_weight),
        "give_margin": round(rng.uniform(0, 1.5) * rollout_weight),
        "backoff_iterations": round(math.exp(rng.uniform(math.log(1), math.log(2048)))),
        "steal_chiplet_penalty": round(rng.uniform(0, 3) * rollout_weight),
        "steal_topk": rng.randint(1, 8),
        "steal_temperature": round(temperature * rollout_weight),
    }


def field(output, name):
    for line in output.splitlines():
        if line.startswith(name + ":"):
            return float(line.split(":", 1)[1].split()[0].rstrip("%"))
    return float("nan")


def mean_wait(output):
    waits = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) > 2 and parts[0].isdigit() and parts[1].endswith("%"):
            waits.append(float(parts[1].rstrip("%")))
    return statistics.mean(waits) if waits else float("nan")


def run(args, config, test):
    # the trial's settings on top of the NANA_CONFIG the script was started with
    path = os.path.join(args.out, "config.json")
    with open(path, "w") as f:
        json.dump({**args.base_config, **config}, f)
    env = dict(os.environ, NANA_CONFIG=path)
    out = subprocess.run([args.bench, str(args.threads), str(args.ms), str(test)],
                         env=env, capture_output=True, text=True, check=True).stdout
    requests = field(out, "requests")
    return {
        "sims_per_s": field(out, "simulations / second"),
        "wait_pct": mean_wait(out),
        "answered_pct": 100 * field(out, "answered") / requests if requests else float("nan"),
        "late": field(out, "late answers"),
    }


def evaluate(args, config, reps, writer, label):
    per_test = []
    waits = []
    for test in (0, 1):
        sims = []
        for rep in range(reps):
            r = run(args, config, test)
            sims.append(r["sims_per_s"])
            waits.append(r["wait_pct"])
            writer.writerow({"label": label, "test": test, "rep": rep, **config, **r})
        per_test.append(statistics.median(sims))
    return math.sqrt(per_test[0] * per_test[1]), statistics.mean(waits)


def key(config):
    return " ".join(f"{k}={config[k]}" for k in DEFAULTS)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bench", required=True)
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--reps", type=int, default=2, help="repetitions per test in the search phase")
    p.add_argument("--confirm", type=int, default=5, help="how many of the best trials to re-run")
    p.add_argument("--confirm-reps", type=int, default=4)
    p.add_argument("--baseline-every", type=int, default=5)
    p.add_argument("--ms", type=int, default=3000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default=f"results/tune-{time.strftime('%Y%m%d-%H%M%S')}")
    args = p.parse_args()
    args.bench = os.path.abspath(args.bench)
    args.base_config = {}
    if os.environ.get("NANA_CONFIG"):
        with open(os.environ["NANA_CONFIG"]) as f:
            args.base_config = json.load(f)

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    fields = ["label", "test", "rep", *DEFAULTS, "sims_per_s", "wait_pct", "answered_pct", "late"]
    log = open(os.path.join(args.out, "runs.csv"), "w", newline="")
    writer = csv.DictWriter(log, fieldnames=fields)
    writer.writeheader()

    baselines = []
    trials = []
    for i in range(args.trials):
        if i % args.baseline_every == 0:
            score, _ = evaluate(args, DEFAULTS, args.reps, writer, "baseline")
            baselines.append((i, score))
            print(f"[{i:3d}] baseline {score / 1e3:9.1f}k", flush=True)
        config = sample(rng)
        score, wait = evaluate(args, config, args.reps, writer, f"trial{i}")
        trials.append((i, config, score, wait))
        log.flush()
        print(f"[{i:3d}] {score / 1e3:9.1f}k  wait {wait:5.1f}%  {key(config)}", flush=True)

    def relative(i, score):
        nearest = min(baselines, key=lambda b: abs(b[0] - i))
        return score / nearest[1]

    ranked = sorted(trials, key=lambda t: relative(t[0], t[2]), reverse=True)

    confirmed = []
    for i, config, _, _ in ranked[:args.confirm]:
        base, base_wait = evaluate(args, DEFAULTS, args.confirm_reps, writer, "confirm-baseline")
        score, wait = evaluate(args, config, args.confirm_reps, writer, f"confirm{i}")
        confirmed.append((score / base, config, wait, base_wait))
        log.flush()
        print(f"confirm trial {i}: x{score / base:.3f} vs defaults  wait {wait:.1f}% (defaults {base_wait:.1f}%)  {key(config)}", flush=True)
    confirmed.sort(key=lambda c: c[0], reverse=True)

    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        def say(line=""):
            print(line)
            f.write(line + "\n")
        say(f"threads {args.threads}, {args.trials} trials x {args.reps} reps x 2 tests, {args.ms} ms per run")
        say(f"baseline (defaults: {key(DEFAULTS)}): " + ", ".join(f"{s / 1e3:.0f}k" for _, s in baselines))
        say("\nsearch phase, top 10 (relative to the nearest baseline):")
        for i, config, score, wait in ranked[:10]:
            say(f"  x{relative(i, score):.3f}  {score / 1e3:8.1f}k  wait {wait:5.1f}%  {key(config)}")
        say(f"\nconfirmed ({args.confirm_reps} reps x 2 tests each, interleaved with the defaults):")
        for ratio, config, wait, base_wait in confirmed:
            say(f"  x{ratio:.3f}  wait {wait:5.1f}% (defaults {base_wait:5.1f}%)  {key(config)}")
    print(f"results in {args.out}")


if __name__ == "__main__":
    sys.exit(main())
