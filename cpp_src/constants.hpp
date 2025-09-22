#pragma once

namespace ofc {
    constexpr int NUM_SUITS = 4;
    constexpr int NUM_RANKS = 13;
    
    // Каналы признаков для представления состояния игры
    constexpr int NUM_FEATURE_CHANNELS = 16; // 8 карт + 5 улиц + 3 скаляра
    constexpr int INFOSET_SIZE = NUM_FEATURE_CHANNELS * NUM_SUITS * NUM_RANKS;

    constexpr int ACTION_VECTOR_SIZE = 208;
}
