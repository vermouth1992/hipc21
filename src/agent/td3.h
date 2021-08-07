//
// Created by Chi Zhang on 8/5/21.
//

#ifndef HIPC21_TD3_H
#define HIPC21_TD3_H

#include "nn/functional.h"
#include "nn/layers.h"
#include "off_policy_agent.h"
#include "gym/gym.h"
#include "common.h"
#include <cmath>


class TD3Agent : public OffPolicyAgent {
public:
    explicit TD3Agent(const Gym::Space &obs_space,
                      const Gym::Space &act_space,
                      int64_t policy_mlp_hidden = 128,
                      float policy_lr = 3e-4,
                      int64_t q_mlp_hidden = 256,
                      float q_lr = 3e-4,
                      float tau = 5e-3,
                      float gamma = 0.99,
                      float actor_noise = 0.1,
                      float target_noise = 0.2,
                      float noise_clip = 0.5
    );

    void update_target_policy(bool soft);

    void log_tabular() override;

    str_to_tensor train_step(const torch::Tensor &obs, const torch::Tensor &act, const torch::Tensor &next_obs,
                             const torch::Tensor &rew, const torch::Tensor &done,
                             const std::optional<torch::Tensor> &importance_weights, bool update_target) override;

    torch::Tensor act_single(const torch::Tensor &obs, bool exploration) override;

private:
    const float actor_noise;
    const float target_noise;
    const float noise_clip;
    const float act_lim;
    torch::nn::AnyModule policy_net;
    torch::nn::AnyModule target_policy_net;
    std::unique_ptr<torch::optim::Optimizer> policy_optimizer;

    torch::Tensor compute_target_q(const torch::Tensor &next_obs,
                                   const torch::Tensor &rew,
                                   const torch::Tensor &done);

    void update_q_net(const torch::Tensor &obs, const torch::Tensor &act, const torch::Tensor &next_obs,
                      const torch::Tensor &rew, const torch::Tensor &done);

    void update_actor(const torch::Tensor &obs);

};


#endif //HIPC21_TD3_H
