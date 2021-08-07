//
// Created by Chi Zhang on 8/7/21.
//

#ifndef HIPC21_OFF_POLICY_TRAINER_H
#define HIPC21_OFF_POLICY_TRAINER_H


#include <utility>

#include "agent/off_policy_agent.h"
#include "replay_buffer/replay_buffer.h"
#include "logger.h"
#include "gym/gym.h"
#include "utils/stop_watcher.h"

class OffPolicyTrainer {
public:
    explicit OffPolicyTrainer(std::shared_ptr<Gym::Environment> env,
                              std::shared_ptr<Gym::Environment> test_env,
                              std::shared_ptr<OffPolicyAgent> agent,
                              const int64_t epochs,
                              const int64_t steps_per_epoch,
                              const int64_t start_steps,
                              const int64_t update_after,
                              const int64_t update_every,
                              const int64_t update_per_step,
                              const int64_t policy_delay,
                              const int64_t num_test_episodes,
                              const torch::Device device,
                              const int64_t seed) :
            env(std::move(env)),
            test_env(std::move(test_env)),
            agent(std::move(agent)),
            epochs(epochs),
            steps_per_epoch(steps_per_epoch),
            num_test_episodes(num_test_episodes),
            start_steps(start_steps),
            update_after(update_after),
            update_every(update_every),
            update_per_step(update_per_step),
            policy_delay(policy_delay),
            device(device),
            seed(seed) {

    }

    void setup_logger(std::optional<std::string> exp_name, const std::string &data_dir) {
        // setup logger
        if (exp_name == std::nullopt) {
            exp_name.emplace(env->env_id + "_" + std::string(typeid(*agent).name()));
        }
        auto output_dir = setup_logger_kwargs(exp_name.value(), seed, data_dir);
        logger = std::make_shared<EpochLogger>(output_dir, exp_name.value());
        agent->set_logger(logger);
    }

    virtual void setup_replay_buffer(int64_t replay_size, int64_t batch_size) {
        std::unique_ptr<DataSpec> action_data_spec;
        auto action_space = env->action_space();
        auto observation_space = env->observation_space();
        if (action_space->type == action_space->DISCRETE) {
            action_data_spec = std::make_unique<DataSpec>(std::vector<int64_t>(), torch::kInt64);
        } else {
            action_data_spec = std::make_unique<DataSpec>(action_space->box_shape, torch::kFloat32);
        }
        // setup agent
        agent->to(device);
        // setup replay buffer
        ReplayBuffer::str_to_dataspec data_spec{
                {"obs",      DataSpec(observation_space->box_shape, torch::kFloat32)},
                {"act",      *action_data_spec},
                {"next_obs", DataSpec(observation_space->box_shape, torch::kFloat32)},
                {"rew",      DataSpec({}, torch::kFloat32)},
                {"done",     DataSpec({}, torch::kFloat32)},
        };

        this->buffer = std::make_shared<UniformReplayBuffer>(replay_size, data_spec, batch_size);
    }

    virtual void train() {
        torch::manual_seed(seed);

        StopWatcher watcher;
        watcher.start();
        env->reset(&s);
        total_steps = 0;
        episode_rewards = 0;
        episode_length = 0;
        num_updates = 0;

        for (int epoch = 1; epoch <= epochs; epoch++) {
            for (int step = 0; step < steps_per_epoch; step++) {
                train_step();
                total_steps += 1;
            }

            // test the current policy
            for (int i = 0; i < num_test_episodes; ++i) {
                test_step();
            }

            watcher.lap();

            // perform logging
            logger->log_tabular("Epoch", epoch);
            logger->log_tabular("EpRet", std::nullopt, true);
            logger->log_tabular("EpLen", std::nullopt, false, true);
            logger->log_tabular("TotalEnvInteracts", (float) total_steps);
            logger->log_tabular("TestEpRet", std::nullopt, true);
            logger->log_tabular("TestEpLen", std::nullopt, false, true);
            agent->log_tabular();
            logger->log_tabular("Time", (float) watcher.seconds());
            logger->dump_tabular();
        }
    }


protected:
    std::shared_ptr<ReplayBuffer> buffer;
    const std::shared_ptr<Gym::Environment> env;
    const std::shared_ptr<Gym::Environment> test_env;
    std::shared_ptr<EpochLogger> logger;
    const std::shared_ptr<OffPolicyAgent> agent;
    const int64_t epochs;
    const int64_t steps_per_epoch;
    const int64_t num_test_episodes;
    const int64_t start_steps;
    const int64_t update_after;
    const int64_t update_every;
    const int64_t update_per_step;
    const int64_t policy_delay;
    const torch::Device device;
    const torch::Device cpu = torch::kCPU;

    int64_t seed;
    int64_t total_steps{};
    float episode_rewards{};
    float episode_length{};
    int64_t num_updates{};
    Gym::State s;

private:
    void test_step() {
        Gym::State test_s;
        // testing variables
        test_env->reset(&test_s);
        float test_episode_reward = 0;
        float test_episode_length = 0;
        while (true) {
            auto tensor_action = agent->act_single(test_s.observation.to(device), false).to(cpu);
            test_env->step(tensor_action, false, &test_s);
            test_episode_reward += test_s.reward;
            test_episode_length += 1;
            if (test_s.done) break;
        }
        logger->store("TestEpRet", test_episode_reward);
        logger->store("TestEpLen", test_episode_length);
    }

    void train_step() {
        // compute action
        torch::Tensor action;
        // copy observation
        auto current_obs = s.observation;
        if (total_steps < start_steps) {
            action = env->action_space()->sample();
        } else {
            action = agent->act_single(current_obs.to(device), true).to(cpu);
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
        buffer->add_single({
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
                    auto idx = buffer->generate_idx();
                    // retrieve the actual data
                    auto data = *buffer->operator[](*idx);
                    // training
                    agent->train_step(data["obs"].to(device),
                                      data["act"].to(device),
                                      data["next_obs"].to(device),
                                      data["rew"].to(device),
                                      data["done"].to(device),
                                      std::nullopt,
                                      num_updates % policy_delay == 0);
                    num_updates += 1;
                }
            }

        }
    }

};


#endif //HIPC21_OFF_POLICY_TRAINER_H
