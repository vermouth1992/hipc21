include_directories(include "${TORCH_INCLUDE_DIRS}")

pybind11_add_module(sumtree pybinding.cpp)
target_link_libraries(sumtree PRIVATE fmt::fmt)

add_executable(env_bind main.cpp)
target_link_libraries(env_bind PRIVATE pybind11::embed)