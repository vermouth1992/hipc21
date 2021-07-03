cmake_minimum_required(VERSION 3.15)
project(hipc21)

set(CMAKE_CXX_STANDARD 14)

set(DEVICE cu111)

set(LIBTORCH /home/chi/Downloads/libtorch-cxx11-abi-shared-with-deps-1.9.0+${DEVICE}/libtorch)

set(CMAKE_PREFIX_PATH "${LIBTORCH}")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


include_directories("./" "${LIBTORCH}/include" "envs/binding-cpp")

add_executable(hipc21 envs/binding-cpp/gym_binding.cpp dqn.h functional.h replay_buffer.h main.cpp)

target_link_libraries(hipc21 "${TORCH_LIBRARIES}" jsoncpp curl m)
set_property(TARGET hipc21 PROPERTY CXX_STANDARD 14)