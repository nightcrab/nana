import re
import os
import subprocess
from collections import defaultdict
num_threads = 4
print("running perf with", num_threads, "threads")
subprocess.run(["sudo","perf", "record", "-F", "30000", "-g", "./nana_bench", str(num_threads), "15000", "1"])

print("generating perf script output")
with open('out.perf', 'w') as file:
    subprocess.run(["sudo","perf","script"], stdout=file)


print("checking if flamegraph dir exists")
if(not os.path.exists("FlameGraph")):
    subprocess.run(["git", "clone", "https://github.com/brendangregg/FlameGraph"])

# Adjusted regex to match your perf output
tid_re = re.compile(r'^\s*\S+\s+(\d+)\s+\d+\.\d+:')  

samples = defaultdict(list)
current_sample = []
current_tid = None

# --- Step 1: Read and split perf script output by thread ---
with open('out.perf', 'r') as f:
    for line in f:
        match = tid_re.match(line)
        if match:
            if current_sample and current_tid:
                samples[current_tid].append(''.join(current_sample))
            current_sample = [line]
            current_tid = match.group(1)
        else:
            current_sample.append(line)

# Save final sample
if current_sample and current_tid:
    samples[current_tid].append(''.join(current_sample))

# Check for threads found
if not samples:
    print("No threads found. Is the perf script empty or incorrectly formatted?")
    exit(1)

print(f"✅ Found {len(samples)} threads with samples.")

# Paths to flamegraph scripts
STACK_COLLAPSE = "./FlameGraph/stackcollapse-perf.pl"
FLAMEGRAPH = "./FlameGraph/flamegraph.pl"

# Check that the scripts exist
for script in [STACK_COLLAPSE, FLAMEGRAPH]:
    if not os.path.isfile(script):
        print(f"Required script '{script}' not found. Make sure you're in the Flamegraph directory or symlink them.")
        exit(1)
# make director for temp files
if not os.path.exists("temp"):
    os.makedirs("temp")

if not os.path.exists("flamegraphs"):
    os.makedirs("flamegraphs")

if not os.path.exists(f"flamegraphs/{num_threads}"):
    os.makedirs(f"flamegraphs/{num_threads}")

# --- Step 2: Write per-thread perf files and generate flamegraphs ---
for tid, traces in samples.items():
    tmp_folder = "temp"
    svg_folder = "flamegraphs"
    base = f"tmp_thread_{tid}"
    perf_file = f"{tmp_folder}/{base}.perf"
    folded_file = f"{tmp_folder}/{base}.folded"
    svg_file = f"{svg_folder}/{num_threads}/{base}.svg"

    # Write .perf file
    with open(perf_file, 'w') as f:
        f.writelines(traces)
    print(f"Wrote {len(traces)} samples to {perf_file}")

    # Run stackcollapse-perf.pl
    try:
        with open(folded_file, 'w') as out_folded:
            subprocess.run([STACK_COLLAPSE, perf_file], check=True, stdout=out_folded)
    except subprocess.CalledProcessError as e:
        print(f"Error during stackcollapse for TID {tid}: {e}")
        continue

    # Run flamegraph.pl
    try:
        with open(svg_file, 'w') as out_svg:
            subprocess.run([FLAMEGRAPH, folded_file], check=True, stdout=out_svg)
        print(f"Generated {svg_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating flamegraph for TID {tid}: {e}")
