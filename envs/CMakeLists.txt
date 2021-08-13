include_directories(include "${LIBTORCH}/include")

set(LIBSOURCE
        binding-cpp/gym_binding.cpp
        )

add_library(gym_cpp ${LIBSOURCE})
target_link_libraries(gym_cpp "${TORCH_LIBRARIES}" curl fmt::fmt)