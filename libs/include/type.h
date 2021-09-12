//
// Created by Chi Zhang on 9/10/21.
//

#ifndef HIPC21_TYPE_H
#define HIPC21_TYPE_H

#include <torch/torch.h>
#include <string>
#include <unordered_map>

namespace rlu {
    struct DataSpec {
        torch::Dtype m_dtype;
        std::vector<int64_t> m_shape;

        DataSpec(std::vector<int64_t> shape, torch::Dtype dtype) : m_dtype(dtype), m_shape(std::move(shape)) {

        }
    };

    typedef std::unordered_map<std::string, torch::Tensor> str_to_tensor;
    typedef std::unordered_map<std::string, torch::autograd::variable_list> str_to_tensor_list;
    typedef std::unordered_map<std::string, DataSpec> str_to_dataspec;

}

#endif //HIPC21_TYPE_H
