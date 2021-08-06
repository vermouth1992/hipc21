//
// Created by Chi Zhang on 8/5/21.
//

#ifndef HIPC21_LINEAR_H
#define HIPC21_LINEAR_H

#include <torch/torch.h>

class EnsembleLinearImpl : public torch::nn::Cloneable<EnsembleLinearImpl> {
public:
    explicit EnsembleLinearImpl(int64_t num_ensembles, int64_t in_features, int64_t out_features, bool use_bias = true);

    void reset_parameters();

    torch::Tensor forward(const torch::Tensor &x);

    void reset() override;

private:
    torch::Tensor weight;
    torch::Tensor bias;
    int64_t in_features;
    int64_t out_features;
    int64_t num_ensembles;
    bool use_bias;
};

TORCH_MODULE(EnsembleLinear);


#endif //HIPC21_LINEAR_H
