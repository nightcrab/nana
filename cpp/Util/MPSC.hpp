#pragma once

#include "SPSC.hpp"
#include <memory>
#include <thread>
#include <vector>

template<class T>
class mpsc {
public:
    explicit mpsc(size_t num_threads) {
        size = num_threads;
        for (int i = 0; i < num_threads; i++) {
            queues.push_back(std::make_unique<rigtorp::SPSCQueue<T>>(1024));
        }
    }
    ~mpsc() = default;
    // non copyable or movable
    mpsc(const mpsc&) = delete;
    mpsc& operator=(const mpsc&) = delete;

    // producer function
    inline void enqueue(T&& data, size_t id) noexcept {
        queues[id]->push(std::move(data));
    }

    // consumer function
    // only touches the producer queues when the local buffer has run dry
    inline bool isempty() noexcept {
        if (flushed_head == flushed_queue.size())
            flush();
        return flushed_head == flushed_queue.size();
    }

    // consumer function
    // jobs already pulled into the local buffer and not yet dequeued (no flush)
    inline size_t buffered() const noexcept {
        return flushed_queue.size() - flushed_head;
    }

    // consumer function
    // drain every producer queue into the local buffer in one pass.
    // the buffer is reset once it's used up; the vector keeps its capacity,
    // so this doesn't allocate once warmed up
    // (std::deque allocated a block per element for large T on MSVC)
    inline void flush() {
        if (flushed_head == flushed_queue.size()) {
            flushed_queue.clear();
            flushed_head = 0;
        }
        for (auto& queue : queues) {
            while (T* item = queue->front()) {
                flushed_queue.emplace_back(std::move(*item));
                queue->pop();
            }
        }
    }

    // consumer function
    inline T dequeue() noexcept {
        while (flushed_head == flushed_queue.size())
            flush();

        return std::move(flushed_queue[flushed_head++]);
    }

    std::vector<T> flushed_queue;
    size_t flushed_head = 0;
    size_t size;
private:
    std::vector<std::unique_ptr<rigtorp::SPSCQueue<T>>> queues;
};
