#include "DeepMCCFR.hpp"
#include "constants.hpp"
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <map>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <atomic>
#include <sstream>

namespace ofc {

// ... (константы и action_to_vector без изменений) ...
const size_t FIRST_STREET_CANDIDATES = 2000;
const size_t FIRST_STREET_ACTION_LIMIT = 100;
const size_t FIRST_STREET_RANDOM_EXPLORE = 5;

std::vector<float> action_to_vector(const Action& action) {
    std::vector<float> vec(ACTION_VECTOR_SIZE, 0.0f);
    const auto& placements = action.first;
    const auto& discarded_card = action.second;
    for (const auto& p : placements) {
        const auto& card = p.first;
        const auto& row_name = p.second.first;
        int slot_idx = -1;
        if (row_name == "top") slot_idx = 0;
        else if (row_name == "middle") slot_idx = 1;
        else if (row_name == "bottom") slot_idx = 2;
        if (slot_idx != -1 && card != INVALID_CARD) {
            vec[card * 4 + slot_idx] = 1.0f;
        }
    }
    if (discarded_card != INVALID_CARD) {
        vec[discarded_card * 4 + 3] = 1.0f;
    }
    return vec;
}

void add_dirichlet_noise(std::vector<float>& strategy, float alpha, std::mt19937& rng) {
    if (strategy.empty()) { return; }
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    std::vector<float> noise(strategy.size());
    float noise_sum = 0.0f;
    for (size_t i = 0; i < strategy.size(); ++i) {
        noise[i] = gamma(rng);
        noise_sum += noise[i];
    }
    if (noise_sum > 1e-6) {
        const float exploration_fraction = 0.25f;
        for (size_t i = 0; i < strategy.size(); ++i) {
            strategy[i] = (1.0f - exploration_fraction) * strategy[i] + exploration_fraction * (noise[i] / noise_sum);
        }
    }
}


std::atomic<uint64_t> DeepMCCFR::request_id_counter_{0};

DeepMCCFR::DeepMCCFR(size_t action_limit, SharedReplayBuffer* policy_buffer, SharedReplayBuffer* value_buffer,
                     InferenceRequestQueue* request_queue, 
                     // === ИЗМЕНЕНИЕ: Принимаем указатель на массив и его ширину ===
                     float* result_array, size_t result_row_size,
                     LogQueue* log_queue) 
    : policy_buffer_(policy_buffer), 
      value_buffer_(value_buffer),
      request_queue_(request_queue),
      // === ИЗМЕНЕНИЕ: Инициализируем новые члены класса ===
      result_array_(result_array),
      result_row_size_(result_row_size),
      log_queue_(log_queue),
      action_limit_(action_limit),
      rng_(std::random_device{}()),
      dummy_action_vec_(ACTION_VECTOR_SIZE, 0.0f)
{}

void DeepMCCFR::run_traversal() {
    GameState state; 
    traverse(state, 0, true);
    state.reset(); 
    traverse(state, 1, true);
}

// ... (featurize_state_cpp без изменений) ...
std::vector<float> DeepMCCFR::featurize_state_cpp(const GameState& state, int player_view) {
    std::vector<float> features(INFOSET_SIZE, 0.0f);
    const int P_BOARD_TOP = 0, P_BOARD_MID = 1, P_BOARD_BOT = 2, P_HAND = 3;
    const int O_BOARD_TOP = 4, O_BOARD_MID = 5, O_BOARD_BOT = 6;
    const int P_DISCARDS = 7, DECK_REMAINING = 8;
    const int IS_STREET_1 = 9, IS_STREET_2 = 10, IS_STREET_3 = 11, IS_STREET_4 = 12, IS_STREET_5 = 13;
    const int O_DISCARD_COUNT = 14, TURN = 15;
    const int plane_size = NUM_SUITS * NUM_RANKS;
    auto set_card = [&](int channel, Card card) {
        if (card != INVALID_CARD) {
            int suit = get_suit(card);
            int rank = get_rank(card);
            features[channel * plane_size + suit * NUM_RANKS + rank] = 1.0f;
        }
    };
    const Board& my_board = state.get_player_board(player_view);
    const Board& opp_board = state.get_opponent_board(player_view);
    for (Card c : my_board.top) set_card(P_BOARD_TOP, c);
    for (Card c : my_board.middle) set_card(P_BOARD_MID, c);
    for (Card c : my_board.bottom) set_card(P_BOARD_BOT, c);
    for (Card c : state.get_dealt_cards()) set_card(P_HAND, c);
    for (Card c : opp_board.top) set_card(O_BOARD_TOP, c);
    for (Card c : opp_board.middle) set_card(O_BOARD_MID, c);
    for (Card c : opp_board.bottom) set_card(O_BOARD_BOT, c);
    for (Card c : state.get_my_discards(player_view)) set_card(P_DISCARDS, c);
    std::vector<bool> known_cards(52, false);
    auto mark_known = [&](Card c) { if (c != INVALID_CARD) known_cards[c] = true; };
    for (Card c : my_board.get_all_cards()) mark_known(c);
    for (Card c : opp_board.get_all_cards()) mark_known(c);
    for (Card c : state.get_dealt_cards()) mark_known(c);
    for (Card c : state.get_my_discards(player_view)) mark_known(c);
    for (int c = 0; c < 52; ++c) {
        if (!known_cards[c]) {
            set_card(DECK_REMAINING, c);
        }
    }
    int street = state.get_street();
    if (street >= 1 && street <= 5) {
        int street_channel = IS_STREET_1 + (street - 1);
        std::fill(features.begin() + street_channel * plane_size, features.begin() + (street_channel + 1) * plane_size, 1.0f);
    }
    float opp_discard_val = static_cast<float>(state.get_opponent_discard_count(player_view)) / 4.0f;
    std::fill(features.begin() + O_DISCARD_COUNT * plane_size, features.begin() + (O_DISCARD_COUNT + 1) * plane_size, opp_discard_val);
    if (state.get_current_player() == player_view) {
        std::fill(features.begin() + TURN * plane_size, features.begin() + (TURN + 1) * plane_size, 1.0f);
    }
    return features;
}


std::map<int, float> DeepMCCFR::traverse(GameState& state, int traversing_player, bool is_root) {
    if (state.is_terminal()) {
        auto payoffs = state.get_payoffs(evaluator_);
        return {{0, payoffs.first}, {1, payoffs.second}};
    }

    int current_player = state.get_current_player();
    std::vector<Action> legal_actions;
    
    // ... (логика выбора действий остается без изменений) ...
    if (state.get_street() == 1) {
        std::vector<Action> candidates;
        state.get_first_street_candidates(FIRST_STREET_CANDIDATES, candidates, rng_);
        
        if (candidates.size() <= FIRST_STREET_ACTION_LIMIT) {
            legal_actions = candidates;
        } else {
            std::map<int, int> suit_map_filter;
            GameState canonical_state_filter = state.get_canonical(suit_map_filter);
            std::vector<float> infoset_vec_filter = featurize_state_cpp(canonical_state_filter, current_player);
            std::vector<std::vector<float>> canonical_action_vectors_filter;
            canonical_action_vectors_filter.reserve(candidates.size());
            for (const auto& original_action : candidates) {
                Action canonical_action = original_action;
                for (auto& placement : canonical_action.first) {
                    if (placement.first != INVALID_CARD) {
                        auto it = suit_map_filter.find(get_suit(placement.first));
                        if (it != suit_map_filter.end()) {
                            placement.first = get_rank(placement.first) * 4 + it->second;
                        }
                    }
                }
                canonical_action_vectors_filter.push_back(action_to_vector(canonical_action));
            }

            // === ИЗМЕНЕНИЕ: Используем кольцевой буфер для ID запросов ===
            uint64_t filter_request_id = (request_id_counter_++) % MAX_PENDING_REQUESTS;
            
            {
                py::gil_scoped_acquire acquire;
                py::tuple filter_request = py::make_tuple(
                    filter_request_id, true, py::cast(infoset_vec_filter), 
                    py::cast(canonical_action_vectors_filter), py::bool_(true), py::bool_(true)
                );
                request_queue_->attr("put")(filter_request);
            }

            std::vector<float> logits;
            // === ИЗМЕНЕНИЕ: Ждем результат в общей памяти ===
            while(result_array_[filter_request_id * result_row_size_ + 1] == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            logits.assign(&result_array_[filter_request_id * result_row_size_ + 2], 
                          &result_array_[filter_request_id * result_row_size_ + 2 + candidates.size()]);
            result_array_[filter_request_id * result_row_size_ + 1] = 0; // Сбрасываем флаг

            std::vector<size_t> indices(candidates.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
                return logits[a] > logits[b];
            });
            std::vector<size_t> final_indices;
            final_indices.reserve(FIRST_STREET_ACTION_LIMIT);
            for(size_t i = 0; i < std::min((size_t)candidates.size(), FIRST_STREET_ACTION_LIMIT - FIRST_STREET_RANDOM_EXPLORE); ++i) {
                final_indices.push_back(indices[i]);
            }
            std::shuffle(indices.begin(), indices.end(), rng_);
            for(size_t idx : indices) {
                if (final_indices.size() >= FIRST_STREET_ACTION_LIMIT) break;
                if (std::find(final_indices.begin(), final_indices.end(), idx) == final_indices.end()) {
                    final_indices.push_back(idx);
                }
            }
            legal_actions.reserve(final_indices.size());
            for(size_t idx : final_indices) {
                legal_actions.push_back(candidates[idx]);
            }
        }
    } else {
        state.get_later_street_actions(legal_actions, rng_);
    }

    int num_actions = legal_actions.size();
    UndoInfo undo_info;

    if (num_actions <= 1) {
        Action action_to_take = (num_actions == 1) ? legal_actions[0] : Action{{}, INVALID_CARD};
        state.apply_action(action_to_take, traversing_player, undo_info);
        auto result = traverse(state, traversing_player, false);
        state.undo_action(undo_info, traversing_player);
        return result;
    }

    std::map<int, int> suit_map;
    GameState canonical_state = state.get_canonical(suit_map);
    std::vector<float> infoset_vec = featurize_state_cpp(canonical_state, current_player);
    
    std::vector<std::vector<float>> canonical_action_vectors;
    canonical_action_vectors.reserve(num_actions);
    
    for (const auto& original_action : legal_actions) {
        Action canonical_action = original_action;
        for (auto& placement : canonical_action.first) {
            if (placement.first != INVALID_CARD) {
                auto it = suit_map.find(get_suit(placement.first));
                if (it != suit_map.end()) {
                    placement.first = get_rank(placement.first) * 4 + it->second;
                }
            }
        }
        if (canonical_action.second != INVALID_CARD) {
            auto it = suit_map.find(get_suit(canonical_action.second));
            if (it != suit_map.end()) {
                canonical_action.second = get_rank(canonical_action.second) * 4 + it->second;
            }
        }
        canonical_action_vectors.push_back(action_to_vector(canonical_action));
    }
    
    // === ИЗМЕНЕНИЕ: Используем кольцевой буфер для ID запросов ===
    uint64_t policy_request_id = (request_id_counter_++) % MAX_PENDING_REQUESTS;
    uint64_t value_request_id = (request_id_counter_++) % MAX_PENDING_REQUESTS;

    bool is_traverser_turn = (current_player == traversing_player);

    {
        py::gil_scoped_acquire acquire;
        py::tuple policy_request_tuple = py::make_tuple(
            policy_request_id, true, py::cast(infoset_vec), py::cast(canonical_action_vectors), 
            py::bool_(is_traverser_turn), py::bool_(false)
        );
        request_queue_->attr("put")(policy_request_tuple);

        py::tuple value_request_tuple = py::make_tuple(
            value_request_id, false, py::cast(infoset_vec), py::none(), 
            py::bool_(is_traverser_turn), py::bool_(false)
        );
        request_queue_->attr("put")(value_request_tuple);
    }

    std::vector<float> logits;
    float value_baseline = 0.0f;
    
    auto start_time = std::chrono::steady_clock::now();
    const auto timeout = std::chrono::seconds(30);

    // === ИЗМЕНЕНИЕ: Ждем оба результата в общей памяти ===
    while(result_array_[policy_request_id * result_row_size_ + 1] == 0 ||
          result_array_[value_request_id * result_row_size_ + 1] == 0) {
        if (std::chrono::steady_clock::now() - start_time > timeout) {
            {
                py::gil_scoped_acquire acquire;
                std::stringstream ss;
                ss << "[C++ WORKER TIMEOUT] Waiting for inference results timed out. Req IDs: " 
                   << policy_request_id << ", " << value_request_id;
                log_queue_->attr("put")(py::str(ss.str()));
            }
            return {{0, 0.0f}, {1, 0.0f}};
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    logits.assign(&result_array_[policy_request_id * result_row_size_ + 2], 
                  &result_array_[policy_request_id * result_row_size_ + 2 + num_actions]);
    value_baseline = result_array_[value_request_id * result_row_size_ + 0];

    // Сбрасываем флаги
    result_array_[policy_request_id * result_row_size_ + 1] = 0;
    result_array_[value_request_id * result_row_size_ + 1] = 0;

    // ... (остальная логика traverse без изменений) ...
    std::vector<float> strategy(num_actions);
    if (!logits.empty() && logits.size() == num_actions) {
        float max_logit = -std::numeric_limits<float>::infinity();
        for(float l : logits) if(l > max_logit) max_logit = l;
        float sum_exp = 0.0f;
        for (int i = 0; i < num_actions; ++i) {
            strategy[i] = std::exp(logits[i] - max_logit);
            sum_exp += strategy[i];
        }
        if (sum_exp > 1e-6) {
            for (int i = 0; i < num_actions; ++i) strategy[i] /= sum_exp;
        } else {
            std::fill(strategy.begin(), strategy.end(), 1.0f / num_actions);
        }
    } else {
        std::fill(strategy.begin(), strategy.end(), 1.0f / num_actions);
    }

    if (is_root) {
        add_dirichlet_noise(strategy, 0.3f, rng_);
    }

    std::discrete_distribution<int> dist(strategy.begin(), strategy.end());
    int sampled_action_idx = dist(rng_);

    state.apply_action(legal_actions[sampled_action_idx], traversing_player, undo_info);
    auto action_payoffs = traverse(state, traversing_player, false);
    state.undo_action(undo_info, traversing_player);

    auto it = action_payoffs.find(current_player);
    if (it != action_payoffs.end()) {
        if (current_player == traversing_player) {
            float advantage = it->second - value_baseline;
            policy_buffer_->push(infoset_vec, canonical_action_vectors[sampled_action_idx], advantage);
        }
        
        value_buffer_->push(infoset_vec, dummy_action_vec_, it->second);
    }
    return action_payoffs;
}

} // namespace ofc 
