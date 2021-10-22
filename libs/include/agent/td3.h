//
// Created by Chi Zhang on 8/5/21.
//

#ifndef HIPC21_TD3_H
#define HIPC21_TD3_H

#include "nn/functional.h"
#include "nn/layers.h"
#include "off_policy_agent.h"
#include "common.h"
#include <cmath>

namespace rlu::agent {

    class TD3Agent : public OffPolicyAgent {
    public:
        explicit TD3Agent(const std::shared_ptr<gym::space::Space> &obs_space,
                          const std::shared_ptr<gym::space::Space> &act_space,
                          int64_t policy_mlp_hidden = 64,
                          float policy_lr = 1e-3,
                          int64_t q_mlp_hidden = 64,
                          int64_t num_q_ensembles = 2,
                          float q_lr = 1e-3,
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

        torch::Tensor
        compute_priority(const torch::Tensor &obs, const torch::Tensor &act, const torch::Tensor &next_obs,
                         const torch::Tensor &rew, const torch::Tensor &done) override;

        void set_grad(const str_to_tensor_list &grads) override;

        std::pair<str_to_tensor_list, str_to_tensor>
        compute_grad(const torch::Tensor &obs, const torch::Tensor &act, const torch::Tensor &next_obs,
                     const torch::Tensor &rew, const torch::Tensor &done,
                     const std::optional<torch::Tensor> &importance_weights, bool update_target) override;

        void update_step(bool update_target) override;

    private:
        float actor_noise;
        float target_noise;
        float noise_clip;
        float act_lim;
        torch::nn::AnyModule policy_net;
        torch::nn::AnyModule target_policy_net;
        std::unique_ptr<torch::optim::Optimizer> policy_optimizer;

        torch::Tensor compute_target_q(const torch::Tensor &next_obs,
                                       const torch::Tensor &rew,
                                       const torch::Tensor &done);

        str_to_tensor update_q_net(const torch::Tensor &obs, const torch::Tensor &act, const torch::Tensor &next_obs,
                                   const torch::Tensor &rew, const torch::Tensor &done,
                                   const std::optional<torch::Tensor> &importance_weights);

        void update_actor(const torch::Tensor &obs);

    };

}
#endif //HIPC21_TD3_H
