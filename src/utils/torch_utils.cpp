//
// Created by Chi Zhang on 8/6/21.
//

#include "torch_utils.h"

torch::Device get_torch_device(const std::string &device_name) {
    torch::DeviceType device_type;
    if (device_name == "gpu" && torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);
    return device;
}
