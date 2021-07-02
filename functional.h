//
// Created by chi on 7/1/21.
//

#ifndef HIPC21_FUNCTIONAL_H
#define HIPC21_FUNCTIONAL_H

#include <torch/torch.h>


static void hard_update(const torch::nn::Module &target, const torch::nn::Module &source) {
    {
        torch::NoGradGuard no_grad;
        for (int i = 0; i < target.parameters().size(); i++) {
            auto target_param = target.parameters()[i];
            auto param = source.parameters()[i];
            target_param.data().copy_(param.data());
        }
    }
}

static void soft_update(const torch::nn::Module &target, const torch::nn::Module &source, float tau) {
    {
        torch::NoGradGuard no_grad;
        for (int i = 0; i < target.parameters().size(); i++) {
            auto target_param = target.parameters()[i];
            auto param = source.parameters()[i];
            target_param.data().copy_(target_param.data() * (1.0 - tau) + param.data() * tau);
        }
    }
}

struct Mlp : torch::nn::Module {
    torch::nn::Linear linear1;
    torch::nn::Linear linear2;
    torch::nn::Linear linear3;

    Mlp(int input_size, int output_size, int mlp_hidden) :
            linear1(register_module("linear1", torch::nn::Linear(input_size, mlp_hidden))),
            linear2(register_module("linear2", torch::nn::Linear(mlp_hidden, mlp_hidden))),
            linear3(register_module("linear3", torch::nn::Linear(mlp_hidden, output_size))) {
    }

    torch::Tensor forward(torch::Tensor x) {
        x = linear1->forward(x);
        x = torch::nn::functional::relu(x);
        x = linear2->forward(x);
        x = torch::nn::functional::relu(x);
        x = linear3->forward(x);
        return x;
    }
};


#endif //HIPC21_FUNCTIONAL_H
