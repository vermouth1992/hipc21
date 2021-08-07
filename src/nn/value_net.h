//
// Created by Chi Zhang on 8/5/21.
//

#ifndef HIPC21_VALUE_NET_H
#define HIPC21_VALUE_NET_H

#include <torch/torch.h>
#include "functional.h"

class EnsembleMinQNet : public torch::nn::Module {
public:
    const int64_t num_ensembles;

    explicit EnsembleMinQNet(int64_t obs_dim, int64_t act_dim, int64_t mlp_hidden, int64_t num_ensembles = 2,
                             int64_t num_layers = 3);

    torch::Tensor forward(const torch::Tensor &obs, const torch::Tensor &act, bool reduce);

private:
    torch::nn::AnyModule model;
};


#endif //HIPC21_VALUE_NET_H
