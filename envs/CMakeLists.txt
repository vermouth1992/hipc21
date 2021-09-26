include_directories(include "${TORCH_INCLUDE_DIRS}")

set(LIBSOURCE
        binding-cpp/gym_binding.cpp
        )

add_library(gym_cpp SHARED ${LIBSOURCE})
target_link_libraries(gym_cpp "${TORCH_LIBRARIES}" curl fmt::fmt)