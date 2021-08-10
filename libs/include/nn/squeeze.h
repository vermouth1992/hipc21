//
// Created by Chi Zhang on 8/5/21.
//

#ifndef HIPC21_SQUEEZE_H
#define HIPC21_SQUEEZE_H

#include <torch/torch.h>

namespace rlu::nn {
    class SqueezeImpl : public torch::nn::Cloneable<SqueezeImpl> {
    public:
        explicit SqueezeImpl(int64_t dim);

        torch::Tensor forward(const torch::Tensor &x);

        void reset() override;

    private:
        int64_t dim;
    };


    TORCH_MODULE(Squeeze);
}


#endif //HIPC21_SQUEEZE_H
