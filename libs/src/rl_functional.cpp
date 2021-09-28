//
// Created by Chi Zhang on 8/6/21.
//

#include "utils/rl_functional.h"

namespace rlu::functional {
    void hard_update(const torch::nn::Module &target, const torch::nn::Module &source) {
        torch::NoGradGuard no_grad;
        for (uint i = 0; i < target.parameters().size(); i++) {
            auto target_param = target.parameters()[i];
            auto param = source.parameters()[i];
            auto device = target_param.device();
            target_param.data().copy_(param.data().to(device));
        }
    }

    void soft_update(const torch::nn::Module &target, const torch::nn::Module &source, float tau) {
        torch::NoGradGuard no_grad;
        for (uint i = 0; i < target.parameters().size(); i++) {
            auto target_param = target.parameters()[i];
            auto param = source.parameters()[i];
            target_param.data().copy_(target_param.data() * (1.0 - tau) + param.data() * tau);
        }
    }
}


