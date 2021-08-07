//
// Created by Chi Zhang on 8/6/21.
//

#ifndef HIPC21_TORCH_UTILS_H
#define HIPC21_TORCH_UTILS_H

#include <torch/torch.h>

torch::Device get_torch_device(const std::string &device_name);


#endif //HIPC21_TORCH_UTILS_H
