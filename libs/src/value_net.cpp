//
// Created by Chi Zhang on 8/5/21.
//

#include "nn/value_net.h"

namespace rlu::nn {
    EnsembleMinQNet::EnsembleMinQNet(int64_t obs_dim, int64_t act_dim, int64_t mlp_hidden, int64_t num_ensembles,
                                     int64_t num_layers) : num_ensembles(num_ensembles) {
        model = register_module("model", build_mlp(obs_dim + act_dim, 1, mlp_hidden, num_layers,
                                                   "relu", true, std::nullopt, num_ensembles));
    }

    torch::Tensor EnsembleMinQNet::forward(const torch::Tensor &obs, const torch::Tensor &act, bool reduce) {
        torch::Tensor x = torch::cat({obs, act}, -1); // (None, obs_dim + act_dim)
        x = x.unsqueeze(0);
        x = x.repeat({num_ensembles, 1, 1});
        x = model.forward(x); // (num_ensemble, None)
        if (reduce) {
            x = std::get<0>(torch::min(x, 0));
        }
        return x;
    }
}


