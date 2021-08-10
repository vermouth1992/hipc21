//
// Created by Chi Zhang on 8/5/21.
//

#include "nn/stack_sequential.h"

namespace rlu::nn {
    torch::Tensor StackSequentialImpl::forward(torch::Tensor x) {
        return SequentialImpl::forward(x);
    }
};


