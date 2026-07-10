import subprocess
import os

tmp_stdout = str()

nps_per_worker = []
bps_per_worker_per_thread = [[]]
jobs_per_worker_per_thread = [[]]

for thread_count in range(1,15):
    proc = subprocess.run(["./nana_bench", str(thread_count), str(100000//thread_count), "0"], stdout=subprocess.PIPE, text=True)
    output = proc.stdout
    for line in output.split("\n"):
        if "nodes / second per worker: " in line:
            line = line.replace("nodes / second per worker: ", "")
            print(f"{thread_count} threads: {line} nps per worker")
            nps_per_worker.append(int(float(line)))

    bps_per_worker_per_thread.append([])
    for thread in range(thread_count):
        bps_per_worker_per_thread[-1].append(int(open("backprops_" + str(thread) + ".txt").read()))
    
    
    jobs_per_worker_per_thread.append([])
    for thread in range(thread_count):
        jobs_per_worker_per_thread[-1].append(sum(1 for _ in open("bench_" + str(thread) + ".txt")))


print("nps per worker:", nps_per_worker)
print("bps_per_worker_per_thread: ")
for bp_per_worker in bps_per_worker_per_thread:
    print(bp_per_worker)

    
print("jobs_per_worker_per_thread: ")
for job_per_worker in jobs_per_worker_per_thread:
    print(job_per_worker)
