cmake_minimum_required(VERSION 3.16)
project(gym_cpp)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(Python COMPONENTS Interpreter Development)
include_directories(${PYTHON_INCLUDE_DIRS})
# assume the anaconda path is two directories above the python executable
get_filename_component(CONDA_PATH ${Python_EXECUTABLE}/../../ ABSOLUTE)

message(STATUS "anaconda path ${CONDA_PATH}")
message(STATUS "Found Python site-package library at ${Python_SITELIB}")

if (DEFINED ENV{LIBTORCH})
    set(LIBTORCH $ENV{LIBTORCH}/share/cmake)
else ()
    set(LIBTORCH ${Python_SITELIB}/torch/share/cmake)
    if (LINUX)
        message(WARNING "Using Libtorch from conda. The google test may not be working correctly")
    endif (LINUX)
endif ()

set(CMAKE_PREFIX_PATH "${LIBTORCH};${CONDA_PATH}")

find_package(pybind11 REQUIRED)
find_package(fmt REQUIRED)
find_package(Torch REQUIRED)
find_package(cppzmq REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXXTORCH_CXX_FLAGS_FLAGS} -O3 -Wall -Wextra -Wno-unused-parameter")

set(gym_cpp_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

include_directories("${gym_cpp_SOURCE_DIR}/include" "${TORCH_INCLUDE_DIRS}" "${gym_cpp_SOURCE_DIR}/third_party")
link_directories(${CONDA_PATH}/lib)

set(SOURCE
        src/envs/env.cpp
        src/spaces/space.cpp
        src/gym.cpp)

message(STATUS ${gym_cpp_SOURCE_DIR})

add_library(gym_cpp SHARED ${SOURCE})
target_link_libraries(gym_cpp "${TORCH_LIBRARIES}" fmt::fmt cppzmq)

