#pragma once

#include <memory>
#include <vector>

#include "SPSC.hpp"

template <typename T>
struct BufferHolder {
    char buffer1[64];
    T val;
    char buffer2[64];
    BufferHolder(auto... param) : val(std::forward<decltype(param)>(param)...) {}

    T* operator->() { return &val; }
};

template<class T>
class mpsc {
public:
    explicit mpsc(size_t num_threads) {
        size = num_threads;
        for (int i = 0; i < num_threads; i++) {
            queues.push_back(std::make_unique<BufferHolder<rigtorp::SPSCQueue<T>>>(1024));
        }
    }
    ~mpsc() = default;
    // non copyable or movable
    mpsc(const mpsc&) = delete;
    mpsc& operator=(const mpsc&) = delete;

    // producer function
    inline void enqueue(const T& data, size_t id) noexcept {
        (*queues[id])->push(data);
    }
    // consumer function
    inline bool isempty() noexcept {
        if(flushed_queue.empty())
            flush();
        return flushed_queue.empty();
    }

    // consumer function
    inline void flush() {
        for (auto& queue : queues) {
            while (T* item = (*queue)->front()) {
                flushed_queue.emplace_back(*item);
                (*queue)->pop();
            }
        }
    }

    // consumer function
    inline T dequeue() noexcept {
        while(flushed_queue.size() == 0uz)
            flush();

        T ret = std::move(flushed_queue.back());
        flushed_queue.pop_back();
        return std::move(ret);
    }
    std::vector<T> flushed_queue;
    size_t size;
private:
    std::vector<std::unique_ptr<BufferHolder<rigtorp::SPSCQueue<T>>>> queues;
};
