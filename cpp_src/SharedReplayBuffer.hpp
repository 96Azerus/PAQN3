#pragma once

#include <cstddef>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <iostream>
#include <mutex>
#include <chrono>
#include <thread>
#include <cmath>

#include "constants.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace ofc {

struct TrainingSample {
    std::vector<float> infoset;
    std::vector<float> action_vector;
    float target_value;

    TrainingSample() 
        : infoset(INFOSET_SIZE, 0.0f), 
          action_vector(ACTION_VECTOR_SIZE, 0.0f), 
          target_value(0.0f) {}
};

class SharedReplayBuffer {
public:
    SharedReplayBuffer(uint64_t capacity) 
        : capacity_(capacity), head_(0), count_(0)
    {
        buffer_.resize(capacity_);
        unsigned seed = static_cast<unsigned>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + 
                        static_cast<unsigned>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
        rng_ = std::make_unique<std::mt19937>(seed);
        std::cout << "C++: Replay Buffer created with capacity " << capacity << std::endl;
    }

    void push(const std::vector<float>& infoset_vec, const std::vector<float>& action_vec, float target) {
        std::lock_guard<std::mutex> lock(mtx_);
        uint64_t index = head_ % capacity_;
        head_++;

        auto& sample = buffer_[index];
        std::copy(infoset_vec.begin(), infoset_vec.end(), sample.infoset.begin());
        std::copy(action_vec.begin(), action_vec.end(), sample.action_vector.begin());
        sample.target_value = target;
        
        if (count_ < capacity_) {
            count_++;
        }
    }

    // Requirement 1.1: push_value and associated normalization variables are removed.

    bool sample_to_ptr(int batch_size, float* infos_ptr, float* actions_ptr, float* targets_ptr) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (count_ < static_cast<uint64_t>(batch_size)) return false;

        std::uniform_int_distribution<uint64_t> dist(0, count_ - 1);

        for (int i = 0; i < batch_size; ++i) {
            uint64_t sample_idx = dist(*rng_);
            const auto& s = buffer_[sample_idx];
            std::copy(s.infoset.begin(), s.infoset.end(),   infos_ptr   + i * INFOSET_SIZE);
            std::copy(s.action_vector.begin(), s.action_vector.end(), actions_ptr + i * ACTION_VECTOR_SIZE);
            targets_ptr[i] = s.target_value;
        }
        return true;
    }
    
    uint64_t size() {
        std::lock_guard<std::mutex> lock(mtx_);
        return count_;
    }

    uint64_t total_generated() {
        std::lock_guard<std::mutex> lock(mtx_);
        return head_;
    }
    
private:
    std::vector<TrainingSample> buffer_;
    uint64_t capacity_;
    uint64_t head_;
    uint64_t count_;
    std::mutex mtx_;
    std::unique_ptr<std::mt19937> rng_;

    // Requirement 1.1: Normalization variables removed.
};

} // namespace ofc
