#pragma once
#include "card.hpp"
#include "hand_evaluator.hpp"
#include <array>
#include <string>
#include <vector>
#include <numeric>

namespace ofc {

    class Board {
    public:
        std::array<Card, 3> top;
        std::array<Card, 5> middle;
        std::array<Card, 5> bottom;

        Board() {
            top.fill(INVALID_CARD);
            middle.fill(INVALID_CARD);
            bottom.fill(INVALID_CARD);
        }

        inline CardSet get_row_cards(const std::string& row_name) const {
            CardSet cards;
            if (row_name == "top") for(Card c : top) if (c != INVALID_CARD) cards.push_back(c);
            else if (row_name == "middle") for(Card c : middle) if (c != INVALID_CARD) cards.push_back(c);
            else if (row_name == "bottom") for(Card c : bottom) if (c != INVALID_CARD) cards.push_back(c);
            return cards;
        }

        inline CardSet get_all_cards() const {
            CardSet all_cards;
            all_cards.reserve(13);
            for(Card c : top) if (c != INVALID_CARD) all_cards.push_back(c);
            for(Card c : middle) if (c != INVALID_CARD) all_cards.push_back(c);
            for(Card c : bottom) if (c != INVALID_CARD) all_cards.push_back(c);
            return all_cards;
        }

        inline int get_card_count() const {
            return get_all_cards().size();
        }

        inline bool is_foul(const HandEvaluator& evaluator) const {
            if (get_card_count() != 13) return false;
            
            CardSet top_cards = get_row_cards("top");
            CardSet mid_cards = get_row_cards("middle");
            CardSet bot_cards = get_row_cards("bottom");

            HandRank top_rank = evaluator.evaluate(top_cards);
            HandRank mid_rank = evaluator.evaluate(mid_cards);
            HandRank bot_rank = evaluator.evaluate(bot_cards);
            return (mid_rank < bot_rank) || (top_rank < mid_rank);
        }

        inline int get_total_royalty(const HandEvaluator& evaluator) const {
            if (is_foul(evaluator)) return 0;

            CardSet top_cards = get_row_cards("top");
            CardSet mid_cards = get_row_cards("middle");
            CardSet bot_cards = get_row_cards("bottom");

            return evaluator.get_royalty(top_cards, "top") +
                   evaluator.get_royalty(mid_cards, "middle") +
                   evaluator.get_royalty(bot_cards, "bottom");
        }

        // --- ИЗМЕНЕНИЕ: Логика "Фантазии" полностью удалена для упрощения ---
    };
}
