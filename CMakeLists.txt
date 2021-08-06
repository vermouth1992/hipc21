cmake_minimum_required(VERSION 3.15)
project(hipc21)

set(CMAKE_CXX_STANDARD 20)

if (APPLE)
    set(CONDA_PATH $ENV{HOME}/miniforge3)
else ()
    set(CONDA_PATH $ENV{HOME}/miniconda3)
endif (APPLE)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

message(STATUS "Found Python site-package library at ${Python_SITELIB}")

set(LIBTORCH ${Python_SITELIB}/torch/share/cmake)

set(CMAKE_PREFIX_PATH "${LIBTORCH};${CONDA_PATH}")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXXTORCH_CXX_FLAGS_FLAGS} -O3 -Wall -Wextra")

include_directories("src" "${LIBTORCH}/include" "envs/binding-cpp/include" "third_party")
link_directories(${CONDA_PATH}/lib)
#link_directories("/opt/intel/oneapi/mkl/2021.3.0/lib/intel64")

add_subdirectory(Google_tests)

set(SOURCE_FILES
        envs/binding-cpp/gym_binding.cpp
        src/replay_buffer/replay_buffer.cpp
        src/replay_buffer/segment_tree.cpp
        src/agent/dqn.cpp
        main.cpp
        src/replay_buffer/SumTree.cpp
        src/agent/sac.cpp src/agent/dqn.cpp
        src/agent/td3.cpp src/nn/linear.cpp src/nn/linear.h src/utils/scheduler.cpp src/utils/scheduler.h src/nn/functional.cpp src/nn/functional.h src/nn/activation.cpp src/nn/activation.h src/nn/layers.h src/nn/squeeze.cpp src/nn/squeeze.h src/utils/stop_watcher.cpp src/utils/stop_watcher.h src/agent/off_policy_agent.cpp src/nn/value_net.cpp src/nn/value_net.h src/nn/stack_sequential.cpp src/nn/stack_sequential.h)

add_executable(hipc21 ${SOURCE_FILES})

find_package(fmt)
target_link_libraries(hipc21 "${TORCH_LIBRARIES}" curl fmt::fmt ${PYTHON_LIBRARIES})

add_library(lib STATIC src/replay_buffer/SumTree.cpp src/replay_buffer/SumTree.h)

add_subdirectory(third_party/pybind11)
pybind11_add_module(sumtree pybinding.cpp)
target_link_libraries(sumtree PRIVATE lib)
