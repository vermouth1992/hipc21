# GymCpp
This repo contains C++ interface to OpenAI Gym. The C++ interface would benefit developing parallel programs to accelerate reinforcement learning algorithms in C++.
As we know doing parallel programming in Python is hard and less efficient. 
This repo provides multiple ways to interact with OpenAI gym with a unified interface:
- ZeroMQ
  - Pros
    - Support any environments in OpenAI Gym
    - Support instantiation of environments in multiple threads
  - Cons
    - Slow
- Pybind11
  - Pros
    - Support any environments in OpenAI Gym
    - Fast
  - Cons
    - Can't have real parallel running environments due to Python GIL. (This can be fixed by sub-interpreters. Unfortunately, it is not officially supported by pybind11)
- Native C++ support
  - Pros
    - Real parallel environments
    - Fast
  - Cons
    - Limited environments
    

## Dependencies

## Examples

## TODO
