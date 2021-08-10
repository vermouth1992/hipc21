//
// Created by Chi Zhang on 8/5/21.
//

#include "nn/squeeze.h"

namespace rlu::nn {
    SqueezeImpl::SqueezeImpl(int64_t dim) : dim(dim) {

    }

    torch::Tensor SqueezeImpl::forward(const torch::Tensor &x) {
        return x.squeeze(dim);
    }

    void SqueezeImpl::reset() {

    }

}

