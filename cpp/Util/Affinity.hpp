#pragma once

// pin worker threads so each gets its own physical core before any core gets two.
// two workers on SMT siblings share a core and run ~25-30% slower each, and the
// OS scheduler doesn't reliably keep them apart.

// pins the calling thread to the index-th logical cpu in physical-core-first order:
// the first logical cpu of every core, then the second ones, and so on.
// indices past the number of usable cpus wrap around. does nothing if the
// topology can't be read.
void pinThreadToCore(int index);

// number of physical cores this process may run on (0 if unknown)
int physicalCoreCount();

// which last-level (L3) cache the cpu that pinThreadToCore(index) uses shares, e.g. its
// chiplet on AMD parts. moving data between L3 groups is much slower than within one.
// 0 for every index if the topology can't be read.
int cacheGroupOfCore(int index);

// restrict the whole process to one logical cpu, before any search thread starts (threads inherit
// it, and pinThreadToCore only picks cpus the process may use). for running several single-thread
// games side by side, as tuner/ does. false if the OS refused
bool restrictProcessToCpu(int cpu);
