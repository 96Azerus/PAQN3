#include "HandEvaluator.h"
#include "OffsetTable.hxx"
#include "Util.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <utility>
#include <cstring>
#include <mutex> // Для потокобезопасной инициализации

namespace omp {

// ... (константы RANKS и FLUSH_RANKS без изменений) ...
const unsigned HandEvaluator::RANKS[]{0x2000, 0x8001, 0x11000, 0x3a000, 0x91000, 0x176005, 0x366000,
        0x41a013, 0x47802e, 0x479068, 0x48c0e4, 0x48f211, 0x494493};
const unsigned HandEvaluator::FLUSH_RANKS[]{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};

Hand Hand::CARDS[CARD_COUNT];
const Hand Hand::EMPTY(0x3333ull << SUITS_SHIFT, 0);
uint16_t HandEvaluator::LOOKUP[86547]; // Убираем макрос
uint16_t* HandEvaluator::ORIG_LOOKUP = nullptr;
uint16_t HandEvaluator::FLUSH_LOOKUP[FLUSH_LOOKUP_SIZE];
const unsigned HandEvaluator::MAX_KEY = 4 * RANKS[12] + 3 * RANKS[11];

// --- ИЗМЕНЕНИЕ: Потокобезопасная инициализация ---
bool HandEvaluator::is_initialized = false;
std::mutex HandEvaluator::init_mutex;
// --- КОНЕЦ ИЗМЕНЕНИЯ ---

HandEvaluator::HandEvaluator()
{
    omp_assert(is_initialized && "HandEvaluator must be initialized by calling HandEvaluator::initialize() before use.");
}

// --- ИЗМЕНЕНИЕ: Явная, потокобезопасная функция инициализации ---
void HandEvaluator::initialize() {
    std::lock_guard<std::mutex> lock(init_mutex);
    if (!is_initialized) {
        initCardConstants();
        staticInit();
        is_initialized = true;
    }
}
// --- КОНЕЦ ИЗМЕНЕНИЯ ---

void HandEvaluator::staticInit()
{
    // ... (остальной код функции staticInit без изменений) ...
    if (RECALCULATE_PERF_HASH_OFFSETS)
        ORIG_LOOKUP = new uint16_t[MAX_KEY + 1];
    static const unsigned RC = RANK_COUNT;
    unsigned handValue = HIGH_CARD;
    handValue = populateLookup(0, 0, handValue, RC, 0, 0, 0);
    handValue = PAIR;
    for (unsigned r = 0; r < RC; ++r)
        handValue = populateLookup(2ull << 4 * r, 2, handValue, RC, 0, 0, 0);
    handValue = TWO_PAIR;
    for (unsigned r1 = 0; r1 < RC; ++r1)
        for (unsigned r2 = 0; r2 < r1; ++r2)
            handValue = populateLookup((2ull << 4 * r1) + (2ull << 4 * r2), 4, handValue, RC, r2, 0, 0);
    handValue = THREE_OF_A_KIND;
    for (unsigned r = 0; r < RC; ++r)
        handValue = populateLookup(3ull << 4 * r, 3, handValue, RC, 0, r, 0);
    handValue = STRAIGHT;
    handValue = populateLookup(0x1000000001111ull, 5, handValue, RC, RC, RC, 3);
    for (unsigned r = 4; r < RC; ++r)
        handValue = populateLookup(0x11111ull << 4 * (r - 4), 5, handValue, RC, RC, RC, r);
    handValue = FLUSH;
    handValue = populateLookup(0, 0, handValue, RC, 0, 0, 0, true);
    handValue = FULL_HOUSE;
    for (unsigned r1 = 0; r1 < RC; ++r1)
        for (unsigned r2 = 0; r2 < RC; ++r2)
            if (r2 != r1)
                handValue = populateLookup((3ull << 4 * r1) + (2ull << 4 * r2), 5, handValue, RC, r2, r1, RC);
    handValue = FOUR_OF_A_KIND;
    for (unsigned r = 0; r < RC; ++r)
        handValue = populateLookup(4ull << 4 * r, 4, handValue, RC, RC, RC, RC);
    handValue = STRAIGHT_FLUSH;
    handValue = populateLookup(0x1000000001111ull, 5, handValue, RC, 0, 0, 3, true);
    for (unsigned r = 4; r < RC; ++r)
        handValue = populateLookup(0x11111ull << 4 * (r - 4), 5, handValue, RC, 0, 0, r, true);
    if (RECALCULATE_PERF_HASH_OFFSETS) {
        calculatePerfectHashOffsets();
        delete[] ORIG_LOOKUP;
    }
}

void HandEvaluator::initCardConstants()
{
    for (unsigned c = 0; c < CARD_COUNT; ++c) {
        unsigned rank = c / 4, suit = c % 4;
        Hand::CARDS[c] = Hand((1ull << (4 * suit + Hand::SUITS_SHIFT)) + (1ull << Hand::CARD_COUNT_SHIFT)
                              + RANKS[rank], 1ull << ((3 - suit) * 16 + rank));
    }
}

// ... (остальные функции файла без изменений) ...
unsigned HandEvaluator::populateLookup(uint64_t ranks, unsigned ncards, unsigned handValue, unsigned endRank,
                                            unsigned maxPair, unsigned maxTrips, unsigned maxStraight, bool flush)
{
    if (ncards <= 5 && ncards >= (MIN_CARDS < 5 ? MIN_CARDS : 5))
        ++handValue;
    if (ncards >= MIN_CARDS || (flush && ncards >= 5)) {
        unsigned key = getKey(ranks, flush);
        if (flush) {
            FLUSH_LOOKUP[key] = handValue;
        } else if (RECALCULATE_PERF_HASH_OFFSETS) {
            ORIG_LOOKUP[key] = handValue;
        } else {
            omp_assert(LOOKUP[perfHash(key)] == 0 || LOOKUP[perfHash(key)] == handValue);
            LOOKUP[perfHash(key)] = handValue;
        }
        if (ncards == 7)
            return handValue;
    }
    for (unsigned r = 0; r < endRank; ++r) {
        uint64_t newRanks = ranks + (1ull << (4 * r));
        unsigned rankCount = ((newRanks >> (r * 4)) & 0xf);
        if (rankCount == 2 && r >= maxPair)
            continue;
        if (rankCount == 3 && r >= maxTrips)
            continue;
        if (rankCount >= 4)
            continue;
        if (getBiggestStraight(newRanks) > maxStraight)
            continue;
        handValue = populateLookup(newRanks, ncards + 1, handValue, r + 1, maxPair, maxTrips, maxStraight, flush);
    }
    return handValue;
}

unsigned HandEvaluator::getKey(uint64_t ranks, bool flush)
{
    unsigned key = 0;
    for (unsigned r = 0; r < RANK_COUNT; ++r)
        key += ((ranks >> r * 4) & 0xf) * (flush ? FLUSH_RANKS[r] : RANKS[r]);
    return key;
}

unsigned HandEvaluator::getBiggestStraight(uint64_t ranks)
{
    uint64_t rankMask = (0x1111111111111 & ranks) | (0x2222222222222 & ranks) >> 1 | (0x4444444444444 & ranks) >> 2;
    for (unsigned i = 9; i-- > 0; )
        if (((rankMask >> 4 * i) & 0x11111ull) == 0x11111ull)
            return i + 4;
    if ((rankMask & 0x1000000001111) == 0x1000000001111)
        return 3;
    return 0;
}

void HandEvaluator::calculatePerfectHashOffsets()
{
    std::vector<std::pair<size_t,std::vector<size_t>>> rows;
    for (size_t i = 0; i < MAX_KEY + 1; ++i) {
        if (ORIG_LOOKUP[i]) {
            size_t rowIdx = i >> PERF_HASH_ROW_SHIFT;
            if (rowIdx >= rows.size())
                rows.resize(rowIdx + 1);
            rows[rowIdx].second.push_back(i);
        }
    }
    for (size_t i = 0; i < rows.size(); ++i)
        rows[i].first = i;
    std::sort(rows.begin(), rows.end(), [](const std::pair<size_t,std::vector<size_t>>& lhs,
              const std::pair<size_t,std::vector<size_t>> & rhs){
        return lhs.second.size() > rhs.second.size();
    });
    size_t maxIdx = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        size_t offset = 0;
        for (;;++offset) {
            bool ok = true;
            for (auto x : rows[i].second) {
                unsigned val = LOOKUP[(x & PERF_HASH_COLUMN_MASK) + offset];
                if (val && val != ORIG_LOOKUP[x]) {
                    ok = false;
                    break;
                }
            }
            if (ok)
                break;
        }
        PERF_HASH_ROW_OFFSETS[rows[i].first] = (uint32_t)(offset - (rows[i].first << PERF_HASH_ROW_SHIFT));
        for (size_t key : rows[i].second) {
            size_t newIdx = (key & PERF_HASH_COLUMN_MASK) + offset;
            maxIdx = std::max<size_t>(maxIdx, newIdx);
            LOOKUP[newIdx] = ORIG_LOOKUP[key];
        }
    }
    std::cout << "offsets: " << std::endl;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (i % 8 == 0)
            std::cout << std::endl;
        std::cout << std::hex << "0x" << PERF_HASH_ROW_OFFSETS[i] << std::dec << ", ";
    }
    std::cout << std::endl;
    outputTableStats("FLUSH_LOOKUP", FLUSH_LOOKUP, 2, FLUSH_LOOKUP_SIZE);
    outputTableStats("ORIG_LOOKUP", ORIG_LOOKUP, 2, MAX_KEY + 1);
    outputTableStats("LOOKUP", LOOKUP, 2, maxIdx + 1);
    outputTableStats("OFFSETS", PERF_HASH_ROW_OFFSETS, 4, rows.size());
    std::cout << "lookup table size: " << maxIdx + 1 << std::endl;
    std::cout << "offset table size: " << rows.size() << std::endl;
}

void HandEvaluator::outputTableStats(const char* name, const void* p, size_t elementSize, size_t count)
{
    char dummy[64]{};
    size_t totalCacheLines = 0, usedCacheLines = 0, usedElements = 0;
    for (size_t i = 0; i < elementSize * count; i += 64) {
        ++totalCacheLines;
        bool used = false;
        for (size_t j = 0; j < 64 && i + j < elementSize * count; j += elementSize) {
            if (std::memcmp((const char*)p + i + j, dummy, elementSize)) {
                ++usedElements;
                used = true;
            }
        }
        usedCacheLines += used;
    }
    std::cout << name << ": cachelines: " << usedCacheLines << "/" << totalCacheLines
         << "  kbytes: " << usedCacheLines / 16  << "/" << totalCacheLines / 16
         << "  elements: " << usedElements << "/" << count
         << std::endl;
}

}
