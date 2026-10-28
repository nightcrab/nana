# Scripts

Run these commands from the repository root. Scripts are grouped by workflow;
Python helpers and their tests stay together so direct script imports work.

| Directory | Tools |
| --- | --- |
| [benchmarks/](benchmarks/) | Speed and strength sweeps, reports, plots, inference benchmarks, and flamegraphs. |
| [policy/](policy/) | Policy iteration, paired A/B evaluation, and policy/tree-dump checks. |
| [opponent/](opponent/) | Stadium collection, Cold Clear analysis, opponent training and export, and the Rust replay helper. |
| [cloud/](cloud/) | EC2 instance control and remote benchmark execution. |

## Common commands

```sh
# Speed scaling and an interactive plot
bash scripts/benchmarks/speed_scaling.sh 32 5 3000
node scripts/benchmarks/plot_speed.mjs <results-dir>

# Strength batches and reports
bash scripts/benchmarks/strength_batches.sh
python scripts/benchmarks/strength_report.py <batch-dirs>
python scripts/benchmarks/plot_strength.py <batch-dirs>

# Policy orchestration checks (no compiled binaries required)
python scripts/policy/test_policy_iterate.py

# Opponent training checks (NumPy and PyTorch required)
python -m unittest discover -s scripts/opponent -p test_opponent_training.py

# Run a benchmark on EC2; SCRIPT also accepts benchmarks/speed_scaling.sh
SCRIPT=speed_scaling.sh bash scripts/cloud/run_ec2_bench.sh <host> <key> <ref>
```

The Rust helper builds with
`cargo build --release --manifest-path scripts/opponent/opponent_replay/Cargo.toml`
and expects the `cold-clear` checkout beside this repository.
