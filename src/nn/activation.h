//
// Created by Chi Zhang on 8/5/21.
//

#ifndef HIPC21_ACTIVATION_H
#define HIPC21_ACTIVATION_H


#include <torch/torch.h>
#include <string>

// create a general activation module with activation function created via string

class ActivationImpl : public torch::nn::Cloneable<ActivationImpl> {
public:
    explicit ActivationImpl(std::string name);

    torch::Tensor forward(const torch::Tensor &x);

    void reset() override;

private:
    std::string name;
};

TORCH_MODULE(Activation);


#endif //HIPC21_ACTIVATION_H
