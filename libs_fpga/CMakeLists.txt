include_directories(include "${LIBTORCH}/include")

set(LIBSOURCE
        src/segment_tree_fpga.cpp
        src/off_policy_trainer_fpga.cpp)

add_library(rlu_fpga SHARED ${LIBSOURCE})
target_link_libraries(rlu_fpga "${TORCH_LIBRARIES}" fmt::fmt rlu)