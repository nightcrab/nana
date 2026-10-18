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
