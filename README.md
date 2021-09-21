# High Performance Parallel Reinforcement Learning Implementation in C++

## Dependencies

We manage all the dependencies using Anaconda for the most simplicity

- nlohmann_json
- spdlog
- fmt
- curl
- pybind11

```bash
conda install nlohmann_json spdlog fmt curl pybind11 flask -c conda-forge
```

If need GPU support

- cudatoolkit-dev (If Pytorch is installed with Cuda support)
- CuDNN

```bash
conda install cudnn cudatoolkit-dev==11.1.1 -c conda-forge
```

- Pytorch

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia 
```