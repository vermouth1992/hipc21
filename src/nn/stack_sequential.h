//
// Created by Chi Zhang on 8/5/21.
//

#ifndef HIPC21_STACK_SEQUENTIAL_H
#define HIPC21_STACK_SEQUENTIAL_H

#include <torch/torch.h>

class StackSequentialImpl : public torch::nn::SequentialImpl {
public:
    using SequentialImpl::SequentialImpl;

    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(StackSequential);

#endif //HIPC21_STACK_SEQUENTIAL_H
