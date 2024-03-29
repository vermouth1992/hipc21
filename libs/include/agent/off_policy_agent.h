//
// Created by Chi Zhang on 8/1/21.
//

#ifndef HIPC21_OFF_POLICY_AGENT_H
#define HIPC21_OFF_POLICY_AGENT_H

#include "torch/torch.h"
#include <optional>
#include <utility>
#include "gym_cpp/spaces/space.h"
#include "logger.h"
#include "type.h"
#include "replay_buffer/replay_buffer_base.h"
#include "utils/rl_functional.h"
#include "utils/stop_watcher.h"

namespace rlu::agent {
    // define a template class for general off-policy agent
    class OffPolicyAgent : public torch::nn::Module {
    public:
        explicit OffPolicyAgent(std::shared_ptr<gym::space::Space> obs_space,
                                std::shared_ptr<gym::space::Space> act_space,
                                float tau, float q_lr, float gamma);

        void update_target_q(bool soft);

        void set_logger(const std::shared_ptr<rlu::logger::EpochLogger> &logger);

        virtual void log_tabular() = 0;

        // train one step
        virtual str_to_tensor train_step(const torch::Tensor &obs,
                                         const torch::Tensor &act,
                                         const torch::Tensor &next_obs,
                                         const torch::Tensor &rew,
                                         const torch::Tensor &done,
                                         const std::optional<torch::Tensor> &importance_weights,
                                         bool update_target) = 0;

        virtual torch::Tensor compute_priority(const torch::Tensor &obs,
                                               const torch::Tensor &act,
                                               const torch::Tensor &next_obs,
                                               const torch::Tensor &rew,
                                               const torch::Tensor &done) = 0;

        // API for parallel training with the parameter server. The agent is the parameter server.
        // compute the gradient and store the info in a dictionary
        virtual std::pair<str_to_tensor_list, str_to_tensor> compute_grad(const torch::Tensor &obs,
                                                                          const torch::Tensor &act,
                                                                          const torch::Tensor &next_obs,
                                                                          const torch::Tensor &rew,
                                                                          const torch::Tensor &done,
                                                                          const std::optional<torch::Tensor> &importance_weights,
                                                                          bool update_target) = 0;

        // set the gradients to the model. Aggregation of the gradients should be done in the trainer
        virtual void set_grad(const str_to_tensor_list &grads) = 0;

        virtual void update_step(bool update_target) = 0;

        virtual torch::Tensor act_single(const torch::Tensor &obs, bool exploration) = 0;

    protected:
        const std::shared_ptr<gym::space::Space> obs_space;
        const std::shared_ptr<gym::space::Space> act_space;
        torch::nn::AnyModule q_network;
        torch::nn::AnyModule target_q_network;
        float tau{};
        float q_lr{};
        float gamma{};
        std::unique_ptr<torch::optim::Optimizer> q_optimizer;
        std::shared_ptr<rlu::logger::EpochLogger> m_logger;
    };
}

#endif //HIPC21_OFF_POLICY_AGENT_H
