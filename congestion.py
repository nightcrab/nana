import subprocess
import os

tmp_stdout = str()

nps_per_worker = []

for thread_count in range(1,15):
    proc = subprocess.run(["./nana_bench", str(thread_count), str(100000//thread_count), "0"], stdout=subprocess.PIPE, text=True)
    output = proc.stdout
    for line in output.split("\n"):
        if "nodes / second per worker: " in line:
            line = line.replace("nodes / second per worker: ", "")
            print(f"{thread_count} threads: {line} nps per worker")
            nps_per_worker.append(int(float(line)))


print("nps per worker:", nps_per_worker)