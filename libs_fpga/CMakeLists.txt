cmake_minimum_required(VERSION 3.15)
project(hipc21)

set(CMAKE_CXX_STANDARD 20)

include_directories(include)
include_directories(${TORCH_INCLUDE_DIRS})
include_directories($ENV{XILINX_XRT_INCLUDE})
include_directories($ENV{XILINX_VIVADO_INCLUDE})

set(CMAKE_PREFIX_PATH "$ENV{XILINX_VIVADO_INCLUDE}/..;$ENV{XILINX_XRT_INCLUDE}/..;${CMAKE_PREFIX_PATH}")

message(STATUS "The Xilinx XRT include distribution $ENV{XILINX_XRT_INCLUDE}")
message(STATUS "The Xilinx VIVADO include distribution $ENV{XILINX_VIVADO_INCLUDE}")

set(LIBSOURCE
        src/segment_tree_fpga.cpp
        src/off_policy_trainer_fpga.cpp)

add_library(rlu_fpga SHARED ${LIBSOURCE})
link_directories("${XILINX_VIVADO}/lib/")
target_link_libraries(rlu_fpga "${TORCH_LIBRARIES}" fmt::fmt rlu OpenCL)