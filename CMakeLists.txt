cmake_minimum_required(VERSION 3.15)
project(hipc21)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(FPGA "Turn on FPGA related code build" OFF)

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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXXTORCH_CXX_FLAGS_FLAGS} -O3 -Wall -Wextra -Wno-unused-parameter")


add_subdirectory(Google_tests)

add_subdirectory(gym_cpp)

include_directories("libs/include" "libs_fpga/include" "${TORCH_INCLUDE_DIRS}" "${gym_cpp_SOURCE_DIR}/include" "third_party")
link_directories(${CONDA_PATH}/lib)

add_subdirectory(libs)

if (FPGA)
    message(STATUS "Build with FPGA support")
    add_definitions(-DUSE_FPGA)
    add_subdirectory(libs_fpga)
    include_directories($ENV{XILINX_XRT_INCLUDE} $ENV{XILINX_VIVADO_INCLUDE})
else ()
    message(STATUS "Build without FPGA support")
endif ()
add_subdirectory(binding)

set(SOURCE_FILES
        src/main.cpp
        )

add_executable(hipc21 ${SOURCE_FILES})

target_link_libraries(hipc21 "${TORCH_LIBRARIES}" fmt::fmt ${PYTHON_LIBRARIES} rlu gym_cpp)
if (FPGA)
    target_link_libraries(hipc21 rlu_fpga)
endif ()

