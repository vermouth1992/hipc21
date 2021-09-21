# High Performance Parallel Reinforcement Learning Implementation in C++

## Dependencies

We manage all the dependencies using Anaconda for the most simplicity

- nlohmann_json
- spdlog
- fmt
- curl
- pybind11
- flask

```bash
conda install nlohmann_json spdlog fmt curl pybind11 flask -c conda-forge
```

- Install Pytorch from [here](https://pytorch.org/get-started/locally/)

If your Pytorch is installed with GPU support, you also need to install the following packages

- cudatoolkit-dev
- CuDNN

```bash
conda install cudnn cudatoolkit-dev==${CUDA_VERSION} -c conda-forge
```

Make sure the CUDA_VERSION matches the one you installed your Pytorch

## Build
Before building, make sure your Python is pointing to the Anaconda environment using
```bash
which python
```
Then, execute
```bash
mkdir build;
cd build;
cmake ..
make -j 8
```

## Running
