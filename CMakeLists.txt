cmake_minimum_required(VERSION 3.15)
project(hipc21)

set(CMAKE_CXX_STANDARD 17)

set(DEVICE cu111)

set(LIBTORCH /home/chi/Downloads/libtorch-cxx11-abi-shared-with-deps-1.9.0+${DEVICE}/libtorch)

set(CMAKE_PREFIX_PATH "${LIBTORCH}")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -fopenmp -O3 -Wall -Wextra")


include_directories("src" "${LIBTORCH}/include" "envs/binding-cpp" "third_party")

add_subdirectory(Google_tests)

set(SOURCE_FILES
        envs/binding-cpp/gym_binding.cpp
        src/replay_buffer/replay_buffer.cpp
        src/replay_buffer/segment_tree.cpp
        src/dqn/dqn.cpp
        main.cpp
        src/replay_buffer/SumTree.cpp)

add_executable(hipc21 ${SOURCE_FILES})

target_link_libraries(hipc21 "${TORCH_LIBRARIES}" jsoncpp curl m)
set_property(TARGET hipc21 PROPERTY CXX_STANDARD 17)


add_library(lib STATIC src/replay_buffer/SumTree.cpp src/replay_buffer/SumTree.h)

add_subdirectory(third_party/pybind11)
pybind11_add_module(sumtree pybinding.cpp)
target_link_libraries(sumtree PRIVATE lib)