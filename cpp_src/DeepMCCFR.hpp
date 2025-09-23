#pragma once

#include <cstddef>
#include <vector>
#include <map>
#include <memory>
#include <random>
#include <atomic>

#include <pybind11/pybind11.h>
#include "game_state.hpp"
#include "hand_evaluator.hpp"
#include "SharedReplayBuffer.hpp"
#include "InferenceQueue.hpp"

namespace py = pybind11;

namespace ofc {

using LogQueue = py::object;

// === ИЗМЕНЕНИЕ: Константа для размера буфера результатов, должна совпадать с Python ===
constexpr size_t MAX_PENDING_REQUESTS = 200 * 4; // NUM_CPP_WORKERS * 4

class DeepMCCFR {
public:
    DeepMCCFR(size_t action_limit,
              SharedReplayBuffer* policy_buffer, 
              SharedReplayBuffer* value_buffer, 
              InferenceRequestQueue* request_queue,
              // === ИЗМЕНЕНИЕ: Принимаем указатель на массив и его ширину ===
              float* result_array,
              size_t result_row_size,
              LogQueue* log_queue);
    
    void run_traversal();

private:
    HandEvaluator evaluator_;
    SharedReplayBuffer* policy_buffer_;
    SharedReplayBuffer* value_buffer_;
    
    InferenceRequestQueue* request_queue_;
    // === ИЗМЕНЕНИЕ: Храним указатель на массив и его ширину ===
    float* result_array_;
    size_t result_row_size_;
    LogQueue* log_queue_;

    size_t action_limit_;
    std::mt19_937 rng_;
    std::vector<float> dummy_action_vec_;
    
    static std::atomic<uint64_t> request_id_counter_;

    std::map<int, float> traverse(GameState& state, int traversing_player, bool is_root);
    std::vector<float> featurize_state_cpp(const GameState& state, int player_view);
};

}
