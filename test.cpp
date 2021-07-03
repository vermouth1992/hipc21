//
// Created by chi on 7/1/21.
//


#include "dqn.h"
#include "replay_buffer.h"
#include <torch/torch.h>
#include <vector>

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


    int64_t obs_dim = 3;
    int64_t act_dim = 6;
    ReplayBuffer::str_to_dataspec data_spec = {
            {"obs",      DataSpec({obs_dim}, torch::kFloat32)},
            {"act",      DataSpec({}, torch::kInt64)},
            {"next_obs", DataSpec({obs_dim}, torch::kFloat32)},
    };

    UniformReplayBuffer buffer(10, data_spec, 2);
    buffer.reset();
    for (int i = 0; i < 5; ++i) {
        ReplayBuffer::str_to_tensor data = {
                {"obs",      torch::rand({3, obs_dim}, torch::TensorOptions().dtype(torch::kFloat32))},
                {"act",      torch::randint(act_dim, {3}, torch::TensorOptions().dtype(torch::kInt64))},
                {"next_obs", torch::rand({3, obs_dim}, torch::TensorOptions().dtype(torch::kFloat32))},
        };
        buffer.add_batch(data);
        std::cout << buffer.size() << " " << buffer.capacity() << std::endl;
        std::cout << *buffer.sample() << std::endl;
    }

    float rand_num = torch::rand({}).item().toFloat();
    std::cout << rand_num << std::endl;

    std::vector<float> data{1.0, 2.0, 3.0};

    torch::Tensor data_tensor = torch::from_blob(data.data(), {(int64_t) data.size()});
    std::cout << data_tensor << std::endl << data_tensor.sizes() << std::endl;

    return 0;
}