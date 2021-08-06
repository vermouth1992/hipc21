//
// Created by Chi Zhang on 8/5/21.
//

#include "stack_sequential.h"

torch::Tensor StackSequentialImpl::forward(torch::Tensor x) {
    return SequentialImpl::forward(x);
}
