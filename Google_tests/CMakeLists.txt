cmake_minimum_required(VERSION 3.15)

# 'Google_test' is the subproject name
project(Google_tests)

# 'lib' is the folder with Google Test sources
add_subdirectory(lib)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR} "${TORCH_INCLUDE_DIRS}" "../libs/include")

find_package(fmt)
# 'Google_Tests_run' is the target name
# 'test1.cpp tests2.cpp' are source files with tests
add_executable(run_test
        test_segment_tree.cpp
        test_libtorch.cpp
        test_cache_performance.cpp
        test_cpu.cpp test_print.cpp
        test_pthread.cpp
        benchmark/parallel_torch.cpp
        test_py_array.cpp
        benchmark/parallel_replay_buffer.cpp)
target_link_libraries(run_test gtest gtest_main "${TORCH_LIBRARIES}" fmt::fmt ${PYTHON_LIBRARIES} gym_cpp rlu pybind11::embed)