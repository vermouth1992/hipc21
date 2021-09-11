//
// Created by Chi Zhang on 9/10/21.
//

#ifndef HIPC21_TYPE_H
#define HIPC21_TYPE_H

#include <torch/torch.h>
#include <string>
#include <unordered_map>

namespace rlu {
    typedef std::unordered_map<std::string, torch::Tensor> str_to_tensor;
    typedef std::unordered_map<std::string, torch::autograd::variable_list> str_to_tensor_list;
}

#endif //HIPC21_TYPE_H
