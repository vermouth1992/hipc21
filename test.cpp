//
// Created by chi on 7/1/21.
//


#include "dqn.h"
#include "replay_buffer.h"
#include <torch/torch.h>
#include <vector>
#include <valarray>

int main() {
    auto dqn = MlpDQN(10, 3, 32, false, 1e-4, 0.99);
    torch::Tensor obs = torch::rand({100, 10});
    torch::Tensor act = torch::randint(3, {100}, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor next_obs = torch::rand({100, 10});
    torch::Tensor reward = torch::rand({100});
    torch::Tensor done = torch::rand({100});
    dqn.train_step(obs, act, next_obs, reward, done);

    auto shape = std::vector<int64_t>{1, 2, 3};
    shape.insert(shape.begin(), 100);
    torch::Tensor tensor = torch::zeros(shape);

    auto index = torch::randint(100, {20}, torch::TensorOptions().dtype(torch::kInt64));

    std::cout << tensor.sizes() << std::endl;
    std::cout << tensor.index({index}).sizes() << std::endl;

    // set values via indexing
    tensor.index_put_({torch::tensor({1, 2})}, 1);

    std::cout << tensor.index({torch::tensor({1, 2})}) << std::endl;

    tensor.index_put_({torch::indexing::Slice(1, 2)}, 2);

    std::cout << tensor.index({torch::tensor({1, 2})}) << std::endl;

    return 0;
}