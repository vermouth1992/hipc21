//
// Created by Chi Zhang on 8/7/21.
//

#include "trainer/off_policy_trainer.h"

#include <utility>
#include "nameof.hpp"
#include "fmt/ostream.h"

namespace rlu::trainer {

    OffPolicyTrainer::OffPolicyTrainer(std::function<std::shared_ptr<Gym::Environment>()> env_fn,
                                       const std::function<std::shared_ptr<agent::OffPolicyAgent>()> &agent_fn,
                                       int64_t epochs, int64_t steps_per_epoch,
                                       int64_t start_steps, int64_t update_after, int64_t update_every,
                                       int64_t update_per_step, int64_t policy_delay, int64_t num_test_episodes,
                                       torch::Device device, int64_t seed, bool online_test) :
            env_fn(std::move(env_fn)),
            agent_fn(agent_fn),
            env(nullptr),
            test_env(nullptr),
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
            seed(seed),
            online_test(online_test) {
    }

    void OffPolicyTrainer::setup_logger(std::optional<std::string> exp_name, const std::string &data_dir) {
        // setup logger
        spdlog::info("Setting up the logger");
        set_default_exp_name(exp_name);
        auto output_dir = rlu::logger::setup_logger_kwargs(exp_name.value(), seed, data_dir);
        logger = std::make_shared<rlu::logger::EpochLogger>(output_dir, exp_name.value());
        agent->set_logger(logger);
    }

    void OffPolicyTrainer::setup_replay_buffer(int64_t replay_size, int64_t batch_size) {
        spdlog::info("Setting up the replay buffer");
        std::unique_ptr<DataSpec> action_data_spec;
        auto action_space = test_env->action_space();
        auto observation_space = test_env->observation_space();
        if (action_space->type == action_space->DISCRETE) {
            action_data_spec = std::make_unique<DataSpec>(std::vector<int64_t>(), torch::kInt64);
        } else {
            action_data_spec = std::make_unique<DataSpec>(action_space->box_shape, torch::kFloat32);
        }
        // setup replay buffer
        str_to_dataspec data_spec{
                {"obs",      DataSpec(observation_space->box_shape, torch::kFloat32)},
                {"act",      *action_data_spec},
                {"next_obs", DataSpec(observation_space->box_shape, torch::kFloat32)},
                {"rew",      DataSpec({}, torch::kFloat32)},
                {"done",     DataSpec({}, torch::kFloat32)},
        };

        this->buffer = std::make_shared<replay_buffer::PrioritizedReplayBuffer<replay_buffer::SegmentTreeTorch>>(
                replay_size, data_spec, batch_size, 0.6);
        this->temp_buffer = std::make_shared<replay_buffer::UniformReplayBuffer>(batch_size, data_spec, 1);
    }

    void OffPolicyTrainer::train() {
        if (online_test) {
            tester = std::make_shared<rlu::trainer::Tester>(test_env, agent, logger, num_test_episodes, device);
        }
        // setup agent
        agent->to(device);
        watcher.start();
        env->reset(&s);

        this->reset();

        spdlog::info("Start training");

        for (int epoch = 1; epoch <= epochs; epoch++) {
            for (int step = 0; step < steps_per_epoch; step++) {
                train_step();
                total_steps += 1;
            }

            // test the current policy
            if (online_test) {
                tester->run();
            }

            watcher.lap();

            // perform logging
            logger->log_tabular("Epoch", epoch);
            logger->log_tabular("EpRet", std::nullopt, true);
            logger->log_tabular("EpLen", std::nullopt, false, true);
            logger->log_tabular("TotalEnvInteracts", (float) total_steps);
            if (online_test) {
                tester->log_tabular();
            }
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

        // store the data in a temporary buffer
        str_to_tensor single_data{{"obs",      current_obs},
                                  {"act",      action},
                                  {"next_obs", s.observation},
                                  {"rew",      reward_tensor},
                                  {"done",     done_tensor}};
        spdlog::debug("Size of the temporary buffer {}", this->temp_buffer->size());
        this->temp_buffer->add_single(single_data);
        spdlog::debug("Size of the temporary buffer {}", this->temp_buffer->size());
        spdlog::debug("Size of the buffer {}", this->buffer->size());

        if (this->temp_buffer->full()) {
            // if the temporary buffer is full, compute the priority and set
            auto storage = this->temp_buffer->get_storage();
            auto priority = this->agent->compute_priority(storage.at("obs").to(device),
                                                          storage.at("act").to(device),
                                                          storage.at("next_obs").to(device),
                                                          storage.at("rew").to(device),
                                                          storage.at("done").to(device));
            storage["priority"] = priority.cpu();
            spdlog::debug("Size of the buffer {}", this->buffer->size());
            buffer->add_batch(storage);
            spdlog::debug("Size of the buffer {}", this->buffer->size());
            this->temp_buffer->reset();
        }

        episode_rewards += s.reward;
        episode_length += 1;
        // handle terminal case
        if (s.done) {
            logger->store({
                                  {"EpRet", std::vector<float>{episode_rewards}},
                                  {"EpLen", std::vector<float>{(float) episode_length}}
                          });
            spdlog::debug("Finish episode");
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
                    spdlog::debug("Agent training");
                    std::optional<torch::Tensor> importance_weights;
                    if (data.contains("weights")) {
                        importance_weights = data["weights"].to(device);
                        spdlog::debug("importance weights min {}, max {}",
                                      torch::min(importance_weights.value()).item<float>(),
                                      torch::max(importance_weights.value()).item<float>());
                    }
                    auto log = agent->train_step(data["obs"].to(device),
                                                 data["act"].to(device),
                                                 data["next_obs"].to(device),
                                                 data["rew"].to(device),
                                                 data["done"].to(device),
                                                 importance_weights,
                                                 num_updates % policy_delay == 0);
                    // update priority
                    log["idx"] = idx;
                    this->buffer->post_process(log);
                    spdlog::debug("After agent training");
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

    void OffPolicyTrainer::set_default_exp_name(std::optional<std::string> &exp_name) {
        if (exp_name == std::nullopt) {
            auto &r = *agent;
            exp_name.emplace(test_env->env_id + "_" + std::string(NAMEOF_SHORT_TYPE_RTTI(r)));
        }
    }

    void OffPolicyTrainer::setup_environment() {
        // setup env on demand on that child class won't instantiate idle env
        spdlog::info("Setting up the environment");
        this->env = env_fn();
        this->test_env = env_fn();
    }

    OffPolicyTrainer::~OffPolicyTrainer() {
        env->close();
        test_env->close();
    }

}