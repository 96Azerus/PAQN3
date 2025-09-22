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
        // --- ИСПРАВЛЕНИЕ: Возвращаем std::array вместо поврежденных типов ---
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

        // --- ОСТАЛЬНАЯ ЧАСТЬ ФАЙЛА ОСТАЕТСЯ БЕЗ ИЗМЕНЕНИЙ (как в вашем "оригинальном" файле) ---
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

        inline bool qualifies_for_fantasyland(const HandEvaluator& evaluator) const {
            if (is_foul(evaluator)) return false;
            
            CardSet top_cards = get_row_cards("top");

            if (top_cards.size() != 3) return false;
            HandRank hr = evaluator.evaluate(top_cards);
            if (hr.type_str == "Pair") {
                int r0 = get_rank(top_cards[0]), r1 = get_rank(top_cards[1]), r2 = get_rank(top_cards[2]);
                int pair_rank = (r0 == r1 || r0 == r2) ? r0 : r1;
                return pair_rank >= 10; // Ранг 10 - это Дама (Q)
            }
            return hr.type_str == "Trips";
        }

        inline int get_fantasyland_card_count(const HandEvaluator& evaluator) const {
            if (!qualifies_for_fantasyland(evaluator)) return 0;
            
            CardSet top_cards = get_row_cards("top");

            HandRank hr = evaluator.evaluate(top_cards);
            if (hr.type_str == "Trips") return 17;
            if (hr.type_str == "Pair") {
                int r0 = get_rank(top_cards[0]), r1 = get_rank(top_cards[1]), r2 = get_rank(top_cards[2]);
                int pair_rank = (r0 == r1 || r0 == r2) ? r0 : r1;
                if (pair_rank == 10) return 14; // QQ
                if (pair_rank == 11) return 15; // KK
                if (pair_rank == 12) return 16; // AA
            }
            return 0;
        }
    };
}
