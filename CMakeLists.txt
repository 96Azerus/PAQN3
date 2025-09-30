# ===================================================================
# ULTRATHINK OPTIMIZED CMakeLists.txt
# ===================================================================
cmake_minimum_required(VERSION 3.14)
project(OFCEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- ✅ Оптимизация флагов компиляции ---
# Для Release-сборки включаем максимальную оптимизацию (-O3),
# используем все доступные инструкции процессора (-march=native)
# и отключаем assert'ы (-DNDEBUG) для скорости.
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -DNDEBUG")
    message(STATUS "🚀 Using Release build flags with native CPU optimizations.")
# Для Debug-сборки оставляем отладочные символы.
else()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -fno-omit-frame-pointer")
    message(STATUS "🛠️ Using Debug build flags.")
endif()


# --- Зависимости ---
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

# --- Исходные файлы ---
set(SOURCES
    "cpp_src/DeepMCCFR.cpp"
    "cpp_src/game_state.cpp"
    "cpp_src/ompeval/omp/CardRange.cpp"
    "cpp_src/ompeval/omp/CombinedRange.cpp"
    "cpp_src/ompeval/omp/EquityCalculator.cpp"
    "cpp_src/ompeval/omp/HandEvaluator.cpp"
    "pybind_wrapper.cpp"
)

# --- Создание цели (Python модуль) ---
pybind11_add_module(ofc_engine SHARED ${SOURCES})

# --- Настройка путей и линковки ---
target_include_directories(ofc_engine PRIVATE
    "${PROJECT_SOURCE_DIR}/cpp_src"
    "${PROJECT_SOURCE_DIR}/cpp_src/ompeval"
)

target_link_libraries(ofc_engine PRIVATE
    pybind11::module
    ${TORCH_LIBRARIES}
    OpenMP::OpenMP_CXX
    Threads::Threads
)

# --- ✅ Дополнительные опции для производительности ---
# Включаем Link-Time Optimization (LTO) для Release сборок, если компилятор поддерживает
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
    set_property(TARGET ofc_engine PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    message(STATUS "✅ Link-Time Optimization (LTO) enabled for Release build.")
endif()
