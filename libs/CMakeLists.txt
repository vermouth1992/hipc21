include_directories(include "${TORCH_INCLUDE_DIRS}")

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
        src/replay_buffer_base.cpp
        src/segment_tree_base.cpp
        src/SumTree.cpp
        src/dqn.cpp
        src/sac.cpp
        src/dqn.cpp
        src/td3.cpp
        src/off_policy_agent.cpp
        src/off_policy_trainer.cpp
        src/off_policy_trainer_parallel.cpp
        src/logger.cpp
        src/tester.cpp
        src/segment_tree_torch.cpp
        src/segment_tree_cpp.cpp
        src/segment_tree_nary.cpp
        src/segment_tree_cppopt.cpp
        src/uniform_replay_buffer.cpp
        )

add_library(rlu SHARED ${LIBSOURCE})
target_link_libraries(rlu "${TORCH_LIBRARIES}" fmt::fmt)