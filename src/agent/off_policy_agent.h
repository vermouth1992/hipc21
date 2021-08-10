//
// Created by Chi Zhang on 8/1/21.
//

#ifndef HIPC21_OFF_POLICY_AGENT_H
#define HIPC21_OFF_POLICY_AGENT_H

#include "torch/torch.h"
#include <optional>
#include "gym/gym.h"
#include "logger.h"
#include "replay_buffer/replay_buffer.h"
#include "utils/rl_functional.h"
#include "utils/stop_watcher.h"

// define a template class for general off-policy agent
class OffPolicyAgent : public torch::nn::Module {
public:
    typedef std::map<std::string, torch::Tensor> str_to_tensor;

    explicit OffPolicyAgent(float tau, float q_lr, float gamma);

    void update_target_q(bool soft);

    void set_logger(const std::shared_ptr<rlu::EpochLogger> &logger);

    virtual void log_tabular() = 0;

    virtual str_to_tensor train_step(const torch::Tensor &obs,
                                     const torch::Tensor &act,
                                     const torch::Tensor &next_obs,
                                     const torch::Tensor &rew,
                                     const torch::Tensor &done,
                                     const std::optional<torch::Tensor> &importance_weights,
                                     bool update_target) = 0;

    virtual torch::Tensor act_single(const torch::Tensor &obs, bool exploration) = 0;

protected:
    torch::nn::AnyModule q_network;
    torch::nn::AnyModule target_q_network;
    float tau{};
    float q_lr{};
    float gamma{};
    std::unique_ptr<torch::optim::Optimizer> q_optimizer;
    std::shared_ptr<rlu::EpochLogger> m_logger;
};


#endif //HIPC21_OFF_POLICY_AGENT_H
