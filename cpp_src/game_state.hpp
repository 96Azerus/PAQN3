#pragma once
#include "board.hpp"
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <map>
#include <string>

namespace ofc {

    struct UndoInfo {
        Action action;
        int prev_street;
        int prev_current_player;
        CardSet dealt_cards_before_action;
    };

    class GameState {
    public:
        GameState(int num_players = 2, int dealer_pos = -1);
        
        void reset(int dealer_pos = -1);

        inline bool is_terminal() const {
            return street_ > 5 || boards_[0].get_card_count() == 13;
        }

        std::pair<float, float> get_payoffs(const HandEvaluator& evaluator) const;
        
        // --- ИЗМЕНЕНИЕ: Разделение логики генерации действий ---
        void get_first_street_candidates(size_t k, std::vector<Action>& out_actions, std::mt19937& rng) const;
        void get_later_street_actions(std::vector<Action>& out_actions, std::mt19937& rng) const;
        // --- КОНЕЦ ИЗМЕНЕНИЯ ---
        
        void apply_action(const Action& action, int player_view, UndoInfo& undo_info);
        
        void undo_action(const UndoInfo& undo_info, int player_view);
        
        GameState get_canonical(std::map<int, int>& suit_map) const;

        int get_street() const { return street_; }
        int get_current_player() const { return current_player_; }
        const CardSet& get_dealt_cards() const { return dealt_cards_; }
        const Board& get_player_board(int player_idx) const { return boards_[player_idx]; }
        const Board& get_opponent_board(int player_idx) const { return boards_[(player_idx + 1) % num_players_]; }
        int get_dealer_pos() const { return dealer_pos_; }

        const CardSet& get_my_discards(int player_idx) const { return my_discards_[player_idx]; }
        
        int get_opponent_discard_count(int player_idx) const { 
            return (int)my_discards_[(player_idx + 1) % num_players_].size(); 
        }
        
    private:
        void deal_cards();
        
        void generate_all_placements_recursive(
            const CardSet& cards_to_place,
            const std::vector<std::pair<std::string, int>>& available_slots,
            std::vector<int>& current_indices,
            int start_idx,
            int k,
            Card discarded,
            std::vector<Action>& out_actions
        ) const;
        
        void generate_random_placements(const CardSet& cards, Card discarded, std::vector<Action>& actions, size_t limit, std::mt19937& rng) const;

        int num_players_;
        int street_;
        int dealer_pos_;
        int current_player_;
        std::vector<Board> boards_;
        CardSet deck_;
        CardSet dealt_cards_;
        std::vector<CardSet> my_discards_;
    };
}
