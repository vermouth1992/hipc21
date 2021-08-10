//
// Created by Chi Zhang on 8/5/21.
//

#ifndef HIPC21_NN_FUNCTIONAL_H
#define HIPC21_NN_FUNCTIONAL_H

#include <torch/torch.h>
#include "layers.h"

namespace rlu::nn {
    StackSequential build_mlp(int64_t input_dim, int64_t output_dim, int64_t mlp_hidden,
                              int64_t num_layers = 3, const std::string &activation = "relu", bool squeeze = false,
                              const std::optional<std::string> &out_activation = std::nullopt,
                              std::optional<int64_t> num_ensembles = std::nullopt,
                              std::optional<float> dropout = std::nullopt
    );

    std::vector<float> convert_tensor_to_flat_vector(const torch::Tensor &tensor);
}


#endif //HIPC21_NN_FUNCTIONAL_H
