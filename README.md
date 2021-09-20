# High Performance Parallel Reinforcement Learning Implementation in C++

## Dependencies
We manage all the dependencies using Anaconda for the best simplicity
- nlohmann_json
- spdlog
- fmt
- curl
- pybind11

```bash
conda install nlohmann_json spdlog fmt curl pybind11 -c conda-forge
```
- Pytorch
```bash
conda install pytorch torchvision torchaudio cudatoolkit=${CUDA_VERSION} -c pytorch -c nvidia 
```
- cudatoolkit-dev (If Pytorch is installed with Cuda support)
- CuDNN
```bash
conda install cudatoolkit-dev==${CUDA_VERSION} -c conda-forge
```
You need to install CuDNN as well and copy the include files and the shared libraries to the ${CONDA_ENV}/include and ${CONDA_ENV}/lib64