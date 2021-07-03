//
// Created by chi on 7/1/21.
//

#ifndef HIPC21_DQN_H
#define HIPC21_DQN_H

#include <torch/torch.h>
#include <boost/range/combine.hpp>
#include <utility>
#include "functional.h"
#include "include/gym/gym.h"
#include "replay_buffer.h"

class DQN : public torch::nn::Module {
protected:
    torch::nn::AnyModule q_network;
    torch::nn::AnyModule target_q_network;
    float tau{};
    bool double_q{};
    float q_lr{};
    float gamma{};
    std::shared_ptr<torch::optim::Adam> optimizer;
public:
    void update_target(bool soft) {
        if (soft) {
            soft_update(*target_q_network.ptr(), *q_network.ptr(), tau);
        } else {
            hard_update(*target_q_network.ptr(), *q_network.ptr());
        }
    }

    void train_step(const torch::Tensor &obs,
                    const torch::Tensor &act,
                    const torch::Tensor &next_obs,
                    const torch::Tensor &rew,
                    const torch::Tensor &done) {

        // compute target values
        torch::Tensor target_q_values;
        {
            torch::NoGradGuard no_grad;
            target_q_values = this->target_q_network.forward(next_obs); // shape (None, act_dim)

            if (double_q) {
                auto target_actions = std::get<1>(torch::max(this->q_network.forward(next_obs), -1)); // shape (None,)
                target_q_values = torch::gather(target_q_values, 1, target_actions.unsqueeze(1)).squeeze(1);
            } else {
                target_q_values = std::get<0>(torch::max(target_q_values, -1));
            }
            target_q_values = rew + gamma * (1. - done) * target_q_values;
        }
        optimizer->zero_grad();
        auto q_values = this->q_network.forward(obs);
        q_values = torch::gather(q_values, 1, act.unsqueeze(1)).squeeze(1);
        auto loss = torch::mse_loss(q_values, target_q_values);
        AT_ASSERT(!std::isnan(loss.template item<float>()));
        loss.backward();
        optimizer->step();
    }

    torch::Tensor act_batch(const torch::Tensor &obs) {
        auto q_values = this->q_network.forward(obs); // shape (None, act_dim)
        auto act = std::get<1>(torch::max(q_values, -1));
        return act;
    }

    torch::Tensor act_single(const torch::Tensor &obs) {
        auto obs_batch = obs.unsqueeze(0);
        auto act_batch = this->act_batch(obs_batch);
        return act_batch.index({0});
    }
};


class MlpDQN : public DQN {
private:
    int64_t obs_dim;
    int64_t act_dim;
public:
    MlpDQN(int64_t obs_dim, int64_t act_dim, int64_t mlp_hidden, bool double_q, float q_lr, float gamma, float tau) {
        this->q_network = std::make_shared<Mlp>(obs_dim, act_dim, mlp_hidden);
        this->target_q_network = std::make_shared<Mlp>(obs_dim, act_dim, mlp_hidden);
        this->optimizer = std::make_shared<torch::optim::Adam>(q_network.ptr()->parameters(),
                                                               torch::optim::AdamOptions(q_lr));
        this->obs_dim = obs_dim;
        this->act_dim = act_dim;
        this->double_q = double_q;
        this->q_lr = q_lr;
        this->gamma = gamma;
        this->tau = tau;

        register_module("q_network", q_network.ptr());
        register_module("target_q_network", target_q_network.ptr());

        update_target(false);
    }
};

class AtariDQN : public DQN {

};


static void train_dqn(
        const boost::shared_ptr<Gym::Client> &client,
        const std::string &env_id,
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
        int64_t replay_size,
        // agent parameters
        int64_t mlp_hidden,
        bool double_q,
        float gamma,
        float q_lr,
        float tau,
        float epsilon_greedy
) {
    // setup environment
    boost::shared_ptr<Gym::Environment> env = client->make(env_id);
    boost::shared_ptr<Gym::Environment> test_env = client->make(env_id);
    boost::shared_ptr<Gym::Space> action_space = env->action_space();
    boost::shared_ptr<Gym::Space> observation_space = env->observation_space();

    auto obs_dim = (int64_t) observation_space->box_low.size();
    int64_t act_dim = action_space->discreet_n;
    // setup agent
    auto agent = MlpDQN(obs_dim, act_dim, mlp_hidden, double_q, q_lr, gamma, tau);
    // setup replay buffer
    ReplayBuffer::str_to_dataspec data_spec = {
            {"obs",      DataSpec({obs_dim}, torch::kFloat32)},
            {"act",      DataSpec({}, torch::kInt64)},
            {"next_obs", DataSpec({obs_dim}, torch::kFloat32)},
            {"rew",      DataSpec({}, torch::kFloat32)},
            {"done",     DataSpec({}, torch::kFloat32)},
    };

    UniformReplayBuffer buffer(replay_size, data_spec, batch_size);
    // main training loop
    Gym::State s;
    env->reset(&s);
    int64_t total_steps = 0;
    float episode_rewards = 0.;
    int64_t episode_length = 0;
    // testing environment variable
    Gym::State test_s;
    std::vector<float> test_reward_result(num_test_episodes, 0.);
    std::vector<int64_t> test_length_result(num_test_episodes, 0);

    // setup timers
    StopWatcher env_step("env_step");
    StopWatcher actor("actor");
    StopWatcher buffer_insert("buffer_insert");
    StopWatcher buffer_sampler("buffer_sampler");
    StopWatcher learner("learner");


    for (int epoch = 1; epoch <= epochs; epoch++) {
        for (int step = 0; step < steps_per_epoch; step++) {
            // compute action
            actor.start();

            std::unique_ptr<std::vector<float>> action;
            // copy observation
            auto current_obs = s.observation;
            auto obs_tensor = torch::from_blob(current_obs.data(), {(int64_t) current_obs.size()});
            if (total_steps < start_steps) {
                action = std::make_unique<std::vector<float>>(action_space->sample());
            } else {
                // epsilon greedy exploration
                float rand_num = torch::rand({}).item().toFloat();
                if (rand_num > epsilon_greedy) {
                    // take agent action
                    auto tensor_action = agent.act_single(obs_tensor).to(torch::kFloat32);  //
                    std::vector<float> vector_action(tensor_action.data_ptr<float>(),
                                                     tensor_action.data_ptr<float>() + tensor_action.numel());
                    action = std::make_unique<std::vector<float>>(vector_action);
                } else {
                    action = std::make_unique<std::vector<float>>(action_space->sample());
                }
            }
            actor.stop();

            env_step.start();
            // environment step
            env->step(*action, false, &s);

            env_step.stop();

            // TODO: need to see if it is true done or done due to reaching the maximum length.

            buffer_insert.start();
            // convert data type
            auto action_tensor = torch::from_blob(action->data(), {(int64_t) action->size()});
            auto next_obs_tensor = torch::from_blob(s.observation.data(), {(int64_t) s.observation.size()});
            auto reward_tensor = torch::tensor({s.reward});
            auto done_tensor = torch::tensor({s.done}, torch::TensorOptions().dtype(torch::kFloat32));

            // store data to the replay buffer
            buffer.add_single({
                                      {"obs",      obs_tensor},
                                      {"act",      action_tensor},
                                      {"next_obs", next_obs_tensor},
                                      {"rew",      reward_tensor},
                                      {"done",     done_tensor}
                              });
            buffer_insert.stop();

            episode_rewards += s.reward;
            episode_length += 1;
            // handle terminal case
            if (s.done) {
                env_step.start();
                env->reset(&s);
                env_step.stop();
                episode_rewards = 0.;
                episode_length = 0;
            }

            // perform learning
            if (total_steps >= update_after) {
                if (total_steps % update_every == 0) {
                    for (int i = 0; i < update_every * update_per_step; i++) {
                        buffer_sampler.start();
                        auto data = *buffer.sample();
                        buffer_sampler.stop();
                        learner.start();
                        agent.train_step(data["obs"], data["act"], data["next_obs"], data["rew"], data["done"]);
                        if (i % policy_delay == 0) {
                            agent.update_target(true);
                        }
                        learner.stop();
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
            int64_t test_episode_length = 0;
            while (true) {
                auto obs_tensor = torch::from_blob(test_s.observation.data(), {(int64_t) test_s.observation.size()});
                auto tensor_action = agent.act_single(obs_tensor).to(torch::kFloat32);  //
                std::vector<float> vector_action(tensor_action.data_ptr<float>(),
                                                 tensor_action.data_ptr<float>() + tensor_action.numel());
                test_env->step(vector_action, false, &test_s);
                assert(test_s.observation.size() == observation_space->sample().size());
                test_episode_reward += test_s.reward;
                test_episode_length += 1;
                if (test_s.done) break;
            }
            test_reward_result[i] = test_episode_reward;
            test_length_result[i] = test_episode_length;
        }
        // compute mean and std of the test performance
        std::pair<float, float> reward_result = compute_mean_std<>(test_reward_result);
        std::pair<float, float> length_result = compute_mean_std<>(test_length_result);

        std::cout << "Epoch " << epoch << " | " << "AverageTestEpReward " << reward_result.first << " | "
                  << "StdTestEpReward " << reward_result.second << " | " << "AverageTestEpLen " << length_result.first
                  << " | " << "StdTestEpLen " << length_result.second << std::endl;
    }

    // print stop_watcher statistics
//    for (auto &stop_watcher:stop_watchers) {
//        std::cout << stop_watcher->name() << ": " << stop_watcher->nanoseconds() << std::endl;
//    }
    std::cout << env_step.name() << ": " << env_step.seconds() << std::endl;
    std::cout << actor.name() << ": " << actor.seconds() << std::endl;
    std::cout << buffer_insert.name() << ": " << buffer_insert.seconds() << std::endl;
    std::cout << buffer_sampler.name() << ": " << buffer_sampler.seconds() << std::endl;
    std::cout << learner.name() << ": " << learner.seconds() << std::endl;

}

#endif //HIPC21_DQN_H
