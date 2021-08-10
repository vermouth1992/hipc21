include_directories(include "${LIBTORCH}/include")

set(LIBSOURCE
        src/activation.cpp
        src/functional.cpp
        src/linear.cpp
        src/rl_functional.cpp
        src/scheduler.cpp
        src/squeeze.cpp
        src/stack_sequential.cpp
        src/stop_watcher.cpp
        src/torch_utils.cpp
        src/value_net.cpp
        src/replay_buffer.cpp
        src/segment_tree.cpp
        src/SumTree.cpp)

add_library(rlu ${LIBSOURCE})
target_link_libraries(rlu "${TORCH_LIBRARIES}")