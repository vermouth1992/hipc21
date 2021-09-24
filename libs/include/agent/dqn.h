//
// Created by chi on 7/1/21.
//

#ifndef HIPC21_DQN_H
#define HIPC21_DQN_H

#include <torch/torch.h>
#include <utility>
#include "gym/gym.h"
#include "replay_buffer/replay_buffer_base.h"
#include "common.h"
#include "agent/off_policy_agent.h"
#include "utils/rl_functional.h"
#include "utils/torch_utils.h"
#include "nn/functional.h"
#include "cxxopts.hpp"
#include "fmt/ranges.h"

namespace rlu::agent {

    class DQN : public OffPolicyAgent {
    public:
        explicit DQN(const Gym::Space &obs_space,
                     const Gym::Space &act_space,
                     int64_t mlp_hidden = 64,
                     int64_t num_layers = 2,
                     float q_lr = 1e-3,
                     float gamma = 0.99,
                     float tau = 5e-3,
                     bool double_q = false,
                     float epsilon_greedy = 0.2);

        str_to_tensor train_step(const torch::Tensor &obs,
                                 const torch::Tensor &act,
                                 const torch::Tensor &next_obs,
                                 const torch::Tensor &rew,
                                 const torch::Tensor &done,
                                 const std::optional<torch::Tensor> &importance_weights,
                                 bool update_target) override;

        torch::Tensor compute_priority(const torch::Tensor &obs,
                                       const torch::Tensor &act,
                                       const torch::Tensor &next_obs,
                                       const torch::Tensor &rew,
                                       const torch::Tensor &done) override;

        std::pair<str_to_tensor_list, str_to_tensor> compute_grad(const torch::Tensor &obs,
                                                                  const torch::Tensor &act,
                                                                  const torch::Tensor &next_obs,
                                                                  const torch::Tensor &rew,
                                                                  const torch::Tensor &done,
                                                                  const std::optional<torch::Tensor> &importance_weights,
                                                                  bool update_target) override;

        void set_grad(const str_to_tensor_list &grads) override;

        void update_step(bool update_target) override;

        torch::Tensor act_single(const torch::Tensor &obs, bool exploration) override;

        torch::Tensor act_test_single(const torch::Tensor &obs);

        void log_tabular() override;

    protected:
        torch::Tensor compute_next_obs_q(const torch::Tensor &next_obs,
                                         const torch::Tensor &rew,
                                         const torch::Tensor &done);


        int64_t m_act_dim;
        bool m_double_q;
        float m_epsilon_greedy;
    };
}


#endif //HIPC21_DQN_H
