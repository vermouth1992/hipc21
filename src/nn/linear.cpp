//
// Created by Chi Zhang on 8/5/21.
//

#include "linear.h"

EnsembleLinearImpl::EnsembleLinearImpl(int64_t num_ensembles, int64_t in_features, int64_t out_features,
                                       bool use_bias) :
        in_features(in_features),
        out_features(out_features),
        num_ensembles(num_ensembles),
        use_bias(use_bias) {
    this->weight = register_parameter("weight", torch::randn({num_ensembles, in_features, out_features}));
    if (use_bias) {
        this->bias = register_parameter("bias", torch::randn({num_ensembles, 1, out_features}));
    }
    reset_parameters();
}

void EnsembleLinearImpl::reset_parameters() {
    auto fan = this->in_features;
    auto gain = torch::nn::init::calculate_gain(torch::kLeakyReLU, sqrt(5.));
    auto std = gain / sqrt(fan);
    auto bound = sqrt(3.0) * std;
    torch::nn::init::uniform_(weight, -bound, bound);
    if (use_bias) {
        auto fan_in = this->in_features;
        bound = 1 / sqrt(fan_in);
        torch::nn::init::uniform_(bias, -bound, bound);
    }

}

torch::Tensor EnsembleLinearImpl::forward(const torch::Tensor &x) {
    return torch::bmm(x, weight) + bias;
}

void EnsembleLinearImpl::reset() {
    reset_parameters();
}

