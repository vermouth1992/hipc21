//
// Created by chi on 7/4/21.
//

#include <gtest/gtest.h>
#include <torch/torch.h>

torch::Tensor same(const torch::Tensor &tensor) {
    return tensor;
}

TEST(Libtorch, tensor_size) {
    for (int i = 1; i < 10; ++i) {
        torch::Tensor data = torch::rand({i}, torch::TensorOptions().dtype(torch::kFloat64));
        std::cout << sizeof(data.data_ptr()) << " " << data.sizes() << data.dtype() << std::endl;
    }

}

TEST(Libtorch, tensor_return) {
    // test the content of the local variable tensor and the returned tensor
    torch::Tensor tensor = torch::rand({5});
    torch::Tensor tensor1 = same(tensor);

    std::cout << &tensor << " " << &tensor1 << std::endl;
    std::cout << tensor.data_ptr() << " " << tensor1.data_ptr() << std::endl;
}