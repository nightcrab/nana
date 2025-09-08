#pragma once

#include <memory>
#include <deque>

#include "SPSC.hpp"

template<typename T, typename Target>
concept forwardable_to = std::same_as<std::remove_cvref_t<T>, Target> && std::convertible_to<T, Target>;


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
        queues.reserve(num_threads);
        for (int i = 0; i < num_threads; i++) {
            queues.push_back(std::make_unique<BufferHolder<rigtorp::SPSCQueue<T>>>(2048));
        }
    }
    ~mpsc() = default;
    // non copyable or movable
    mpsc(const mpsc&) = delete;
    mpsc& operator=(const mpsc&) = delete;

    // producer function
    template<std::convertible_to<T> Data>
    void enqueue(Data &&data, size_t id) noexcept {
        (*queues[id])->emplace(std::forward<Data>(data));
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
                flushed_queue.emplace_back(std::move(*item));
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

        T ret = std::move(flushed_queue.front());
        flushed_queue.pop_front();
        return std::move(ret);
    }
    
    std::deque<T> flushed_queue;
    size_t size;
private:
    std::vector<std::unique_ptr<BufferHolder<rigtorp::SPSCQueue<T>>>> queues;
};
