//
// Created by Chi Zhang on 8/5/21.
//

#include "nn/activation.h"
#include "fmt/core.h"


namespace rlu::nn {
    ActivationImpl::ActivationImpl(std::string name) : name(std::move(name)) {

    }

    torch::Tensor ActivationImpl::forward(const torch::Tensor &x) {
        if (name == "relu") {
            return x.relu();
        } else if (name == "tanh") {
            return x.tanh();
        } else if (name == "sigmoid") {
            return x.sigmoid();
        } else {
            throw std::runtime_error(fmt::format("Unknown activation {}", name));
        }
    }

    void ActivationImpl::reset() {

    }

}

