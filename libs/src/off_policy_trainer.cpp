//
// Created by Chi Zhang on 8/7/21.
//

#include "trainer/off_policy_trainer.h"
#include "nameof.hpp"

namespace rlu::trainer {

    OffPolicyTrainer::OffPolicyTrainer(const std::function<std::shared_ptr<Gym::Environment>()> &env_fn,
                                       const std::function<std::shared_ptr<agent::OffPolicyAgent>()> &agent_fn,
                                       int64_t epochs, int64_t steps_per_epoch,
                                       int64_t start_steps, int64_t update_after, int64_t update_every,
                                       int64_t update_per_step, int64_t policy_delay, int64_t num_test_episodes,
                                       torch::Device device, int64_t seed) :
            env_fn(env_fn),
            agent_fn(agent_fn),
            env(env_fn()),
            test_env(env_fn()),
            agent(agent_fn()),
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

    void OffPolicyTrainer::setup_logger(std::optional<std::string> exp_name, const std::string &data_dir) {
        // setup logger
        if (exp_name == std::nullopt) {
            auto &r = *agent;
            exp_name.emplace(env->env_id + "_" + std::string(NAMEOF_SHORT_TYPE_RTTI(r)));
        }
        auto output_dir = rlu::logger::setup_logger_kwargs(exp_name.value(), seed, data_dir);
        logger = std::make_shared<rlu::logger::EpochLogger>(output_dir, exp_name.value());
        agent->set_logger(logger);
    }

    void OffPolicyTrainer::setup_replay_buffer(int64_t replay_size, int64_t batch_size) {
        std::unique_ptr<rlu::replay_buffer::DataSpec> action_data_spec;
        auto action_space = env->action_space();
        auto observation_space = env->observation_space();
        if (action_space->type == action_space->DISCRETE) {
            action_data_spec = std::make_unique<rlu::replay_buffer::DataSpec>(std::vector<int64_t>(), torch::kInt64);
        } else {
            action_data_spec = std::make_unique<rlu::replay_buffer::DataSpec>(action_space->box_shape, torch::kFloat32);
        }
        // setup agent
        agent->to(device);
        // setup replay buffer
        str_to_dataspec data_spec{
                {"obs",      DataSpec(observation_space->box_shape, torch::kFloat32)},
                {"act",      *action_data_spec},
                {"next_obs", DataSpec(observation_space->box_shape, torch::kFloat32)},
                {"rew",      DataSpec({}, torch::kFloat32)},
                {"done",     DataSpec({}, torch::kFloat32)},
        };

        this->buffer = std::make_shared<rlu::replay_buffer::UniformReplayBuffer>(replay_size, data_spec, batch_size);
    }

    void OffPolicyTrainer::train() {
        torch::manual_seed(seed);
        watcher.start();
        env->reset(&s);

        this->reset();

        for (int epoch = 1; epoch <= epochs; epoch++) {
            for (int step = 0; step < steps_per_epoch; step++) {
                train_step();
                total_steps += 1;
            }

            // test the current policy
            for (int i = 0; i < num_test_episodes; ++i) {
                test_step(this->agent);
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

    void OffPolicyTrainer::test_step(const std::shared_ptr<agent::OffPolicyAgent> &test_actor) {
        Gym::State test_s;
        // testing variables
        test_env->reset(&test_s);
        float test_episode_reward = 0;
        float test_episode_length = 0;
        while (true) {
            auto tensor_action = test_actor->act_single(test_s.observation.to(device), false).to(cpu);
            test_env->step(tensor_action, false, &test_s);
            test_episode_reward += test_s.reward;
            test_episode_length += 1;
            if (test_s.done) break;
        }
        logger->store("TestEpRet", test_episode_reward);
        logger->store("TestEpLen", test_episode_length);
    }

    void OffPolicyTrainer::train_step() {
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

        //

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
                    auto data = buffer->operator[](idx);
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

    void OffPolicyTrainer::reset() {
        total_steps = 0;
        episode_rewards = 0;
        episode_length = 0;
        num_updates = 0;
    }

}