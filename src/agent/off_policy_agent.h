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

    void set_logger(const std::shared_ptr<EpochLogger> &logger);

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
    std::shared_ptr<EpochLogger> m_logger;
};


// off-policy trainer for low dimensional observation environments
void off_policy_trainer(
        const std::shared_ptr<Gym::Environment> &env,
        const std::shared_ptr<Gym::Environment> &test_env,
        std::optional<std::string> exp_name,
        const std::string &data_dir,
        int64_t epochs,
        int64_t steps_per_epoch,
        int64_t start_steps,
        int64_t update_after,
        int64_t update_every,
        int64_t update_per_step,
        int64_t policy_delay,
        int64_t batch_size,
        int64_t num_test_episodes,
        int64_t seed,
        // replay buffer
        int64_t replay_size,
        // agent parameters
        const std::shared_ptr<OffPolicyAgent> &agent,
        // torch
        torch::Device device
);


#endif //HIPC21_OFF_POLICY_AGENT_H
