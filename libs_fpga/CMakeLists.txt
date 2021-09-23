include_directories(include "${TORCH_INCLUDE_DIRS}")

set(LIBSOURCE
        src/segment_tree_fpga.cpp
        src/off_policy_trainer_fpga.cpp)

add_library(rlu_fpga SHARED ${LIBSOURCE})
link_directories()
target_link_libraries(rlu_fpga "${TORCH_LIBRARIES}" fmt::fmt rlu)