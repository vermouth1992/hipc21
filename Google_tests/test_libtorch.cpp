//
// Created by chi on 7/4/21.
//

#include <gtest/gtest.h>
#include <torch/torch.h>
#include "nlohmann/json.hpp"
#include "nn/functional.h"
#include <fstream>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

using nlohmann::json;

torch::Tensor same(const torch::Tensor &tensor) {
    return tensor;
}

TEST(Libtorch, grad) {
    auto model = rlu::nn::build_mlp(5, 2, 2, 3, "tanh");
    fmt::print("The number of parameters is {}\n\n", model->parameters().size());
    torch::Tensor data = torch::ones({2, 5}, torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor out = torch::sum(model->forward(data));
    auto g = torch::autograd::grad({out}, model->parameters());
    fmt::print("{}\n, grads:\n{}\n, parameters:\n{}\n\n", out, g, model->parameters());
//    out.backward();
    fmt::print("{}\n\n", model->parameters()[0].grad());
    model->parameters()[0].mutable_grad() = g[0];
    fmt::print("{}\n\n", model->parameters()[0].grad());
//    out = torch::sum(data);
//    g = torch::autograd::grad({out}, model->parameters());
//    fmt::print("{}\n", g);
//    out.backward();
//    fmt::print("{}\n", data.grad());
}

TEST(Libtorch, cat) {
    torch::Tensor a = torch::arange(10, 14);
    torch::Tensor b = torch::arange(5);
    torch::Tensor c = torch::cat({a, b}, 0);
    fmt::print("{}\n", c);
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

//
//TEST(Libtorch, serialization) {
//    // read file
//    std::ifstream input("../../tensor.pt");
//    std::string str((std::istreambuf_iterator<char>(input)),
//                    std::istreambuf_iterator<char>());
//    std::string s;
//    // b64decode
//    macaron::Base64::Decode(str, s);
//    std::vector<char> f(s.length());
//    std::copy(s.begin(), s.end(), f.begin());
//    torch::Tensor x = torch::pickle_load(f).toTensor();
//    std::cout << x << std::endl;
//}
//
//TEST(Libtorch, cpp_to_python) {
//    torch::Tensor x = torch::rand({3, 4}, torch::TensorOptions().dtype(torch::kFloat32));
//    std::cout << x << std::endl;
//    std::vector<char> f = torch::pickle_save(x);
//    std::string result(f.begin(), f.end());
////    std::string result = macaron::Base64::Encode(f);
//    std::ofstream out("../../tensor.pt");
//    out << result;
//    out.close();
//}
