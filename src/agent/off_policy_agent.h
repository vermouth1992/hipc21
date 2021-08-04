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
#include "functional.h"

// define a template class for general off-policy agent
class OffPolicyAgent : public torch::nn::Module {
public:
    typedef std::map<std::string, torch::Tensor> str_to_tensor;

    void update_target(bool soft) {
        if (soft) {
            soft_update(*target_q_network.ptr(), *q_network.ptr(), tau);
        } else {
            hard_update(*target_q_network.ptr(), *q_network.ptr());
        }
    }

    void set_logger(const std::shared_ptr<EpochLogger> &logger) {
        this->m_logger = logger;
    }

    virtual void log_tabular() = 0;

    virtual str_to_tensor train_step(const torch::Tensor &obs,
                                     const torch::Tensor &act,
                                     const torch::Tensor &next_obs,
                                     const torch::Tensor &rew,
                                     const torch::Tensor &done,
                                     const std::optional<torch::Tensor> &importance_weights) = 0;

    virtual torch::Tensor act_batch(const torch::Tensor &obs, bool exploration) = 0;

    virtual torch::Tensor act_single(const torch::Tensor &obs, bool exploration) {
        auto obs_batch = obs.unsqueeze(0);
        auto act_batch = this->act_batch(obs_batch, exploration);
        return act_batch.index({0});
    }

protected:
    torch::nn::AnyModule q_network;
    torch::nn::AnyModule target_q_network;
    float tau{};
    float q_lr{};
    float gamma{};
    std::unique_ptr<torch::optim::Adam> q_optimizer;
    std::shared_ptr<EpochLogger> m_logger;
};


// off-policy trainer for low dimensional observation environments
static void off_policy_trainer(
        const std::shared_ptr<Gym::Client> &client,
        const std::string &env_id,
        std::optional<std::string> &exp_name,
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
        OffPolicyAgent &agent,
        // torch
        torch::Device device
) {
    // setup logger
    if (exp_name == std::nullopt) {
        exp_name.emplace(env_id + "_" + std::string(typeid(agent).name()));
    }
    auto output_dir = setup_logger_kwargs(exp_name.value(), seed, data_dir);
    std::shared_ptr<EpochLogger> logger = std::make_shared<EpochLogger>(output_dir, exp_name.value());
    agent.set_logger(logger);

    torch::manual_seed(seed);
    // setup environment
    std::shared_ptr<Gym::Environment> env = client->make(env_id);
    std::shared_ptr<Gym::Environment> test_env = client->make(env_id);
    std::shared_ptr<Gym::Space> action_space = env->action_space();
    std::shared_ptr<Gym::Space> observation_space = env->observation_space();

    std::unique_ptr<DataSpec> action_dataspec;
    if (action_space->type == action_space->DISCRETE) {
        action_dataspec = std::make_unique<DataSpec>(std::vector<int64_t>(), torch::kInt64);
    } else {
        action_dataspec = std::make_unique<DataSpec>(action_space->box_shape, torch::kFloat32);
    }
    // setup agent
    agent.to(device);
    // setup replay buffer
    ReplayBuffer::str_to_dataspec data_spec{
            {"obs",      DataSpec(observation_space->box_shape, torch::kFloat32)},
            {"act",      *action_dataspec},
            {"next_obs", DataSpec(observation_space->box_shape, torch::kFloat32)},
            {"rew",      DataSpec({}, torch::kFloat32)},
            {"done",     DataSpec({}, torch::kFloat32)},
    };

    UniformReplayBuffer buffer(replay_size, data_spec, batch_size);
    // main training loop
    Gym::State s;
    env->reset(&s);
    int64_t total_steps = 0;
    float episode_rewards = 0.;
    float episode_length = 0;
    // testing environment variable
    Gym::State test_s;
    std::vector<float> test_reward_result(num_test_episodes, 0.);
    std::vector<float> test_length_result(num_test_episodes, 0);

    torch::Device cpu(torch::kCPU);

    StopWatcher watcher;
    watcher.start();

    for (int epoch = 1; epoch <= epochs; epoch++) {
        for (int step = 0; step < steps_per_epoch; step++) {
            // compute action
            torch::Tensor action;
            // copy observation
            auto current_obs = s.observation;
            if (total_steps < start_steps) {
                action = action_space->sample();
            } else {
                action = agent.act_single(current_obs.to(device), true).to(cpu).to(torch::kFloat32);
            }

            // environment step
            env->step(action, false, &s);

            // TODO: need to see if it is true done or done due to reaching the maximum length.
            // convert data type
            auto reward_tensor = torch::tensor({s.reward});
            bool true_done = s.done & (!s.timeout);
            auto done_tensor = torch::tensor({true_done},
                                             torch::TensorOptions().dtype(torch::kFloat32));

            // store data to the replay buffer
            buffer.add_single({
                                      {"obs",      current_obs},
                                      {"act",      action},
                                      {"next_obs", s.observation},
                                      {"rew",      reward_tensor},
                                      {"done",     done_tensor}
                              });

            episode_rewards += s.reward;
            episode_length += 1;
            // handle terminal case
            if (s.done) {
                logger->store({
                                      {"EpRet", std::vector<float>{episode_rewards}},
                                      {"EpLen", std::vector<float>{(float) episode_length}}
                              });

                env->reset(&s);
                episode_rewards = 0.;
                episode_length = 0;
            }

            // perform learning
            if (total_steps >= update_after) {
                if (total_steps % update_every == 0) {
                    for (int i = 0; i < update_every * update_per_step; i++) {
                        // generate index
                        auto idx = buffer.generate_idx();
                        // retrieve the actual data
                        auto data = *buffer[*idx];
                        // training
                        agent.train_step(data["obs"].to(device),
                                         data["act"].to(device),
                                         data["next_obs"].to(device),
                                         data["rew"].to(device),
                                         data["done"].to(device),
                                         std::nullopt);
                        if (i % policy_delay == 0) {
                            agent.update_target(true);
                        }
                    }
                }

            }

            total_steps += 1;
        }

        // test the current policy
        for (int i = 0; i < num_test_episodes; ++i) {
            // testing variables
            test_env->reset(&test_s);
            float test_episode_reward = 0;
            float test_episode_length = 0;
            while (true) {
                auto obs_tensor = s.observation;
                auto tensor_action = agent.act_single(obs_tensor.to(device), false).to(cpu).to(torch::kFloat32);  //
                test_env->step(tensor_action, false, &test_s);
                test_episode_reward += test_s.reward;
                test_episode_length += 1;
                if (test_s.done) break;
            }
            test_reward_result[i] = test_episode_reward;
            test_length_result[i] = test_episode_length;
        }
        logger->store({
                              {"TestEpRet", test_reward_result},
                              {"TestEpLen", test_length_result}

                      });

        watcher.lap();

        // perform logging
        logger->log_tabular("Epoch", epoch);
        logger->log_tabular("EpRet", std::nullopt, true);
        logger->log_tabular("EpLen", std::nullopt, false, true);
        logger->log_tabular("TotalEnvInteracts", (float) total_steps);
        logger->log_tabular("TestEpRet", std::nullopt, true);
        logger->log_tabular("TestEpLen", std::nullopt, false, true);
        agent.log_tabular();
        logger->log_tabular("Time", (float) watcher.seconds());
        logger->dump_tabular();
    }

}


#endif //HIPC21_OFF_POLICY_AGENT_H
