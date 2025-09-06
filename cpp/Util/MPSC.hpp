#pragma once

#include <memory>
#include <vector>

#include "SPSC.hpp"

template <typename T>
struct BufferHolder {
    alignas(std::hardware_destructive_interference_size) T val;
    BufferHolder(auto... param) : val(std::forward<decltype(param)>(param)...) {}

    T* operator->() { return &val; }
};

template<class T>
class mpsc {
public:
    explicit mpsc(size_t num_threads) {
        size = num_threads;
        for (int i = 0; i < num_threads; i++) {
            queues.push_back(std::make_unique<BufferHolder<rigtorp::SPSCQueue<T>>>(2048));
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

    inline void clear() {
        for (auto& queue : queues) {
            while (T* item = (*queue)->front()) {
                (*queue)->pop();
            }
        }
        flushed_queue.clear();
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
