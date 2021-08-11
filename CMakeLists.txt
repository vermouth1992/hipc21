cmake_minimum_required(VERSION 3.15)
project(hipc21)

set(CMAKE_CXX_STANDARD 20)

if (APPLE)
    set(CONDA_PATH $ENV{HOME}/miniforge3)
else ()
    set(CONDA_PATH $ENV{HOME}/miniconda3)
endif (APPLE)

find_package(Python COMPONENTS Interpreter Development)
include_directories(${PYTHON_INCLUDE_DIRS})

message(STATUS "Found Python site-package library at ${Python_SITELIB}")

set(LIBTORCH ${Python_SITELIB}/torch/share/cmake)

set(CMAKE_PREFIX_PATH "${LIBTORCH};${CONDA_PATH}")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXXTORCH_CXX_FLAGS_FLAGS} -O3 -Wall -Wextra")

include_directories("libs/include" "src" "${LIBTORCH}/include" "envs/binding-cpp/include" "third_party")
link_directories(${CONDA_PATH}/lib)
#link_directories("/opt/intel/oneapi/mkl/2021.3.0/lib/intel64")

find_package(pybind11)
add_subdirectory(Google_tests)
add_subdirectory(libs)
add_subdirectory(binding)

set(SOURCE_FILES
        envs/binding-cpp/gym_binding.cpp
        src/agent/dqn.cpp
        main.cpp
        src/agent/sac.cpp
        src/agent/dqn.cpp
        src/agent/td3.cpp
        src/agent/off_policy_agent.cpp
        src/trainer/off_policy_trainer.cpp
        src/trainer/off_policy_trainer_parallel.cpp
        )

add_executable(hipc21 ${SOURCE_FILES})

find_package(fmt)
target_link_libraries(hipc21 "${TORCH_LIBRARIES}" curl fmt::fmt ${PYTHON_LIBRARIES} rlu)

