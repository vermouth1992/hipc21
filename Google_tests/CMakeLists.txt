cmake_minimum_required(VERSION 3.15)

# 'Google_test' is the subproject name
project(Google_tests)

# 'lib' is the folder with Google Test sources
add_subdirectory(lib)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR} "${LIBTORCH}/include" "../envs/binding-cpp/include")

find_package(fmt)
# 'Google_Tests_run' is the target name
# 'test1.cpp tests2.cpp' are source files with tests
add_executable(Google_Tests_run
        test_segment_tree.cpp test_libtorch.cpp test_share_ptr.cpp
        test_cache_performance.cpp test_cpu.cpp test_print.cpp test_python.cpp test_env.cpp
        ../envs/binding-cpp/gym_binding.cpp)
target_link_libraries(Google_Tests_run gtest gtest_main "${TORCH_LIBRARIES}" fmt::fmt curl ${PYTHON_LIBRARIES})