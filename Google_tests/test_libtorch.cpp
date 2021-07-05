//
// Created by chi on 7/4/21.
//

#include <gtest/gtest.h>
#include <torch/torch.h>

TEST(Libtorch, tensor_size) {
    for (int i = 1; i < 10; ++i) {
        torch::Tensor data = torch::rand({i}, torch::TensorOptions().dtype(torch::kFloat64));
        std::cout << sizeof(data.data_ptr()) << " " << data.sizes() << data.dtype() << std::endl;
    }

}