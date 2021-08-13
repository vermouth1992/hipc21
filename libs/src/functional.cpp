//
// Created by Chi Zhang on 8/5/21.
//

#include "nn/functional.h"

namespace rlu::nn {
    StackSequential
    build_mlp(int64_t input_dim, int64_t output_dim, int64_t mlp_hidden, int64_t num_layers,
              const std::string &activation,
              bool squeeze, const std::optional<std::string> &out_activation, std::optional<int64_t> num_ensembles,
              std::optional<float> dropout) {
        auto model = StackSequential();
        if (num_layers == 1) {
            if (num_ensembles == std::nullopt) {
                model->push_back(torch::nn::Linear(input_dim, output_dim));
            } else {
                model->push_back(EnsembleLinear(num_ensembles.value(), input_dim, output_dim));
            }
        } else {
            // add first layer
            if (num_ensembles == std::nullopt) {
                model->push_back(torch::nn::Linear(input_dim, mlp_hidden));
            } else {
                model->push_back(EnsembleLinear(num_ensembles.value(), input_dim, mlp_hidden));
            }
            // add activation
            model->push_back(Activation(activation));
            // add dropout
            if (dropout != std::nullopt) {
                model->push_back(torch::nn::Dropout(torch::nn::DropoutOptions().p(dropout.value())));
            }

            // intermediate layers
            for (int i = 0; i < num_layers - 2; i++) {
                if (num_ensembles == std::nullopt) {
                    model->push_back(torch::nn::Linear(mlp_hidden, mlp_hidden));
                } else {
                    model->push_back(EnsembleLinear(num_ensembles.value(), mlp_hidden, mlp_hidden));
                }
                // add activation
                model->push_back(Activation(activation));
                // add dropout
                if (dropout != std::nullopt) {
                    model->push_back(torch::nn::Dropout(torch::nn::DropoutOptions().p(dropout.value())));
                }
            }

            // last layer
            if (num_ensembles == std::nullopt) {
                model->push_back(torch::nn::Linear(mlp_hidden, output_dim));
            } else {
                model->push_back(EnsembleLinear(num_ensembles.value(), mlp_hidden, output_dim));
            }

            // last activation
            if (out_activation != std::nullopt) {
                model->push_back(Activation(out_activation.value()));
            }
            // optional squeeze
            if (output_dim == 1 && squeeze) {
                model->push_back(Squeeze(-1));
            }
        }

        return model;
    }

}


