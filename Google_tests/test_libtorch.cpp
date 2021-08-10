//
// Created by chi on 7/4/21.
//

#include <gtest/gtest.h>
#include <torch/torch.h>
#include "nlohmann/json.hpp"
#include "base64.h"
#include "nn/functional.h"
#include <fstream>

using nlohmann::json;

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


TEST(Libtorch, serialization) {
    // read file
    std::ifstream input("../../tensor.pt");
    std::string str((std::istreambuf_iterator<char>(input)),
                    std::istreambuf_iterator<char>());
    std::string s;
    // b64decode
    macaron::Base64::Decode(str, s);
    std::vector<char> f(s.length());
    std::copy(s.begin(), s.end(), f.begin());
    torch::Tensor x = torch::pickle_load(f).toTensor();
    std::cout << x << std::endl;
}

TEST(Libtorch, cpp_to_python) {
    torch::Tensor x = torch::rand({3, 4}, torch::TensorOptions().dtype(torch::kFloat32));
    std::cout << x << std::endl;
    std::vector<char> f = torch::pickle_save(x);
    std::string result(f.begin(), f.end());
//    std::string result = macaron::Base64::Encode(f);
    std::ofstream out("../../tensor.pt");
    out << result;
    out.close();
}
