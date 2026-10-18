// Route every C++ allocation (operator new/delete) through mimalloc.
//
// The search allocates a lot from many threads at once: nodes, their action vectors,
// move generation buffers, state copies. With glibc's malloc, growing and shrinking the
// per-thread heaps calls mprotect, which takes the process' mmap lock; on a 32-core
// Linux machine that cost ~25% of throughput. mimalloc's thread-local heaps avoid it.
//
// This file must be compiled into each executable exactly once; mimalloc requires
// that its new/delete overrides are defined in a single translation unit.
#ifdef NANA_MIMALLOC
#include <mimalloc-new-delete.h>
#endif
