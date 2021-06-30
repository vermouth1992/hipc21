cmake_minimum_required(VERSION 3.19)
project(hipc21)

set(CMAKE_CXX_STANDARD 14)


include_directories("/opt/homebrew/include/eigen3/")
include_directories("./")

add_executable(hipc21 main.cpp Env.h CartPole.cpp CartPole.h)