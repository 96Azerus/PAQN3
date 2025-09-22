#ifndef OMP_HAND_EVALUATOR_H
#define OMP_HAND_EVALUATOR_H

#include "Util.h"
#include "Constants.h"
#include "Hand.h"
#include <cstdint>
#include <cassert>
#include <mutex> // Для потокобезопасной инициализации

namespace omp {

class HandEvaluator
{
public:
    // --- ИЗМЕНЕНИЕ: Явная, потокобезопасная функция инициализации ---
    static void initialize();

    HandEvaluator();

    template<bool tFlushPossible = true>
    OMP_FORCE_INLINE uint16_t evaluate(const Hand& hand) const
    {
        omp_assert(hand.count() <= 7 && hand.count() == bitCount(hand.mask()));
        if (!tFlushPossible || !hand.hasFlush()) {
            uint32_t key = hand.rankKey();
            return LOOKUP[perfHash(key)];
        } else {
            uint16_t flushKey = hand.flushKey();
            omp_assert(flushKey < FLUSH_LOOKUP_SIZE);
            return FLUSH_LOOKUP[flushKey];
        }
    }

private:
    // --- ИЗМЕНЕНИЕ: Статические члены для управления инициализацией ---
    static bool is_initialized;
    static std::mutex init_mutex;

    static unsigned perfHash(unsigned key)
    {
        omp_assert(key <= MAX_KEY);
        return key + PERF_HASH_ROW_OFFSETS[key >> PERF_HASH_ROW_SHIFT];
    }

    static void initCardConstants();
    static void staticInit();
    static void calculatePerfectHashOffsets();
    static unsigned populateLookup(uint64_t rankCounts, unsigned ncards, unsigned handValue, unsigned endRank,
                                   unsigned maxPair, unsigned maxTrips, unsigned maxStraight, bool flush = false);
    static unsigned getKey(uint64_t rankCounts, bool flush);
    static unsigned getBiggestStraight(uint64_t rankCounts);
    static void outputTableStats(const char* name, const void* p, size_t elementSize, size_t count);

    static const unsigned RANKS[RANK_COUNT];
    static const unsigned FLUSH_RANKS[RANK_COUNT];
    static const bool RECALCULATE_PERF_HASH_OFFSETS = false;
    static const unsigned PERF_HASH_ROW_SHIFT = 12;
    static const unsigned PERF_HASH_COLUMN_MASK = (1 << PERF_HASH_ROW_SHIFT) - 1;
    static const unsigned MIN_CARDS = 0;
    static const unsigned MAX_KEY;
    static const size_t FLUSH_LOOKUP_SIZE = 8192;
    static uint16_t* ORIG_LOOKUP;
    static uint16_t LOOKUP[]; // Размер определяется в .hxx
    static uint16_t FLUSH_LOOKUP[FLUSH_LOOKUP_SIZE];
    static uint32_t PERF_HASH_ROW_OFFSETS[]; // Размер определяется в .hxx
};

}

#endif // OMP_HAND_EVALUATOR_H
