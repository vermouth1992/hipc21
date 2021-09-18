//
// Created by Chi Zhang on 8/6/21.
//

#include "utils/torch_utils.h"
#include "spdlog/spdlog.h"

namespace rlu::ptu {
    torch::Device get_torch_device(const std::string &device_name) {
        torch::DeviceType device_type;
        if (device_name == "cpu") {
            spdlog::info("Training on CPU.");
            device_type = torch::kCPU;
        } else if (device_name == "gpu") {
            if (torch::cuda::is_available()) {
                spdlog::info("CUDA available! Training on GPU.");
                device_type = torch::kCUDA;
            } else {
                spdlog::info("CUDA is not available. Training on CPU.");
                device_type = torch::kCPU;
            }
        } else {
            throw std::runtime_error(fmt::format("Unknown device {}", device_name));
        }
        torch::Device device(device_type);
        return device;
    }
}


