cmake_minimum_required(VERSION 3.15)

# 'Google_test' is the subproject name
project(Google_tests)

# 'lib' is the folder with Google Test sources
add_subdirectory(lib)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR} "${LIBTORCH}/include")

set(DEVICE cu111)

set(LIBTORCH /home/chi/Downloads/libtorch-cxx11-abi-shared-with-deps-1.9.0+${DEVICE}/libtorch)

set(CMAKE_PREFIX_PATH "${LIBTORCH}")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


# 'Google_Tests_run' is the target name
# 'test1.cpp tests2.cpp' are source files with tests
add_executable(Google_Tests_run test_segment_tree.cpp test_libtorch.cpp test_share_ptr.cpp)
target_link_libraries(Google_Tests_run gtest gtest_main "${TORCH_LIBRARIES}")