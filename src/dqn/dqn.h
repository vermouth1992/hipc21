//
// Created by chi on 7/1/21.
//

#ifndef HIPC21_DQN_H
#define HIPC21_DQN_H

#include <torch/torch.h>
#include <utility>
#include "functional.h"
#include "include/gym/gym.h"
#include "replay_buffer/replay_buffer.h"
#include "common.h"

class DQN : public torch::nn::Module {
public:
    typedef std::map<std::string, torch::Tensor> str_to_tensor;

    void update_target(bool soft) {
        if (soft) {
            soft_update(*target_q_network.ptr(), *q_network.ptr(), tau);
        } else {
            hard_update(*target_q_network.ptr(), *q_network.ptr());
        }
    }

    std::shared_ptr<str_to_tensor> train_step(const torch::Tensor &obs,
                                              const torch::Tensor &act,
                                              const torch::Tensor &next_obs,
                                              const torch::Tensor &rew,
                                              const torch::Tensor &done,
                                              const torch::Tensor &importance_weights) {

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
        auto loss = torch::square(q_values - target_q_values); // (None,)
        loss = torch::mean(loss * importance_weights);
        loss.backward();
        optimizer->step();

        str_to_tensor log_data{
                {"abs_delta_q", torch::abs(q_values - target_q_values).detach()}
        };
        return std::make_shared<str_to_tensor>(log_data);
    }

    torch::Tensor act_batch(const torch::Tensor &obs) {
        {
            torch::NoGradGuard no_grad;
            auto q_values = this->q_network.forward(obs); // shape (None, act_dim)
            auto act = std::get<1>(torch::max(q_values, -1));
            return act;
        }
    }

    torch::Tensor act_single(const torch::Tensor &obs) {
        auto obs_batch = obs.unsqueeze(0);
        auto act_batch = this->act_batch(obs_batch);
        return act_batch.index({0});
    }

protected:
    torch::nn::AnyModule q_network;
    torch::nn::AnyModule target_q_network;
    float tau{};
    bool double_q{};
    float q_lr{};
    float gamma{};
    std::shared_ptr<torch::optim::Adam> optimizer;
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
        // replay buffer
        float alpha,
        float initial_beta,
        int64_t replay_size,
        // agent parameters
        int64_t mlp_hidden,
        bool double_q,
        float gamma,
        float q_lr,
        float tau,
        float epsilon_greedy,
        // torch
        torch::Device device
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
    agent.to(device);
    // setup replay buffer
    ReplayBuffer::str_to_dataspec data_spec = {
            {"obs",      DataSpec({obs_dim}, torch::kFloat32)},
            {"act",      DataSpec({}, torch::kInt64)},
            {"next_obs", DataSpec({obs_dim}, torch::kFloat32)},
            {"rew",      DataSpec({}, torch::kFloat32)},
            {"done",     DataSpec({}, torch::kFloat32)},
    };

    PrioritizedReplayBuffer buffer(replay_size, data_spec, batch_size, alpha);
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
    StopWatcher buffer_indexing("buffer_indexing");
    StopWatcher buffer_sampler("buffer_sampler");
    StopWatcher buffer_update_priority("buffer_update_priority");
    StopWatcher learner("learner");

    std::vector<StopWatcher *> stop_watchers{&env_step, &actor, &buffer_insert, &buffer_indexing,
                                             &buffer_sampler, &learner, &buffer_update_priority};

    torch::Device cpu(torch::kCPU);

    LinearSchedule beta_scheduler(epochs * steps_per_epoch, 1.0, initial_beta);

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
                    auto tensor_action = agent.act_single(obs_tensor.to(device)).to(cpu).to(torch::kFloat32);  //
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
            bool true_done = s.done & (!s.timeout);
            auto done_tensor = torch::tensor({true_done}, torch::TensorOptions().dtype(torch::kFloat32));

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
                        // generate index
                        buffer_indexing.start();
                        auto idx = buffer.generate_idx();
                        buffer_indexing.stop();
                        // get importance weights
                        auto weights = buffer.get_weights(*idx, (float) beta_scheduler.value(total_steps));
                        // retrieve the actual data
                        buffer_sampler.start();
                        auto data = *buffer[*idx];
                        buffer_sampler.stop();
                        // training
                        learner.start();
                        auto logs = agent.train_step(data["obs"].to(device),
                                                     data["act"].to(device),
                                                     data["next_obs"].to(device),
                                                     data["rew"].to(device),
                                                     data["done"].to(device),
                                                     weights->to(device));
                        if (i % policy_delay == 0) {
                            agent.update_target(true);
                        }
                        learner.stop();
                        // update priority
                        buffer_update_priority.start();
                        buffer.update_priorities(*idx, (*logs)["abs_delta_q"]);
                        buffer_update_priority.stop();
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
                auto tensor_action = agent.act_single(obs_tensor.to(device)).to(cpu).to(torch::kFloat32);  //
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
    double total = 0.;
    for (auto &stop_watcher:stop_watchers) {
        total += stop_watcher->seconds();
    }
    for (auto &stop_watcher:stop_watchers) {
        std::cout << stop_watcher->name() << ": " << stop_watcher->seconds() / total * 100. << "%" << std::endl;
    }

    std::cout << "Total execution time: " << total << " s" << std::endl;
}


struct ActorParam {
    boost::shared_ptr<Gym::Environment> env;
    std::shared_ptr<PrioritizedReplayBuffer> replay_buffer;
    std::shared_ptr<MlpDQN> agent; // the agent should live on CPU
    std::shared_ptr<pthread_mutex_t> actor_mutex;
    std::shared_ptr<pthread_mutex_t> buffer_mutex;
    // used to determine when to exit
    std::shared_ptr<pthread_mutex_t> global_steps_mutex;
    std::shared_ptr<int64_t> global_steps;
    int64_t max_global_steps;
    int64_t start_steps;
    float epsilon_greedy;

    ActorParam(boost::shared_ptr<Gym::Environment> env_,
               std::shared_ptr<PrioritizedReplayBuffer> replay_buffer_,
               std::shared_ptr<MlpDQN> agent_,
               std::shared_ptr<pthread_mutex_t> actor_mutex_,
               std::shared_ptr<pthread_mutex_t> buffer_mutex_,
               std::shared_ptr<pthread_mutex_t> global_steps_mutex_,
               std::shared_ptr<int64_t> global_steps_,
               int64_t max_global_steps_,
               int64_t start_steps_,
               float epsilon_greedy_
    ) :
            env(std::move(env_)),
            replay_buffer(std::move(replay_buffer_)),
            agent(std::move(agent_)),
            actor_mutex(std::move(actor_mutex_)),
            buffer_mutex(std::move(buffer_mutex_)),
            global_steps_mutex(std::move(global_steps_mutex_)),
            global_steps(std::move(global_steps_)),
            max_global_steps(max_global_steps_),
            start_steps(start_steps_),
            epsilon_greedy(epsilon_greedy_) {

    }
};


void *actor_fn(void *params) {
    std::cout << "Creating actor thread " << pthread_self() << std::endl;
    // each actor has the client,
    auto *actor_params = (ActorParam *) params;
    auto env = actor_params->env;
    auto replay_buffer = actor_params->replay_buffer;
    auto agent = actor_params->agent;
    auto actor_mutex = actor_params->actor_mutex;
    auto global_steps = actor_params->global_steps;
    auto global_steps_mutex = actor_params->global_steps_mutex;
    auto max_global_steps = actor_params->max_global_steps;
    auto epsilon_greedy = actor_params->epsilon_greedy;
    auto start_steps = actor_params->start_steps;
    auto buffer_mutex = actor_params->buffer_mutex;

    boost::shared_ptr<Gym::Space> action_space = env->action_space();
    boost::shared_ptr<Gym::Space> observation_space = env->observation_space();

    int64_t global_steps_temp;
    Gym::State s;
    s.done = true;
    // main loop
    while (true) {
        if (s.done) {
            env->reset(&s);
        }
        std::unique_ptr<std::vector<float>> action;
        // copy observation
        auto current_obs = s.observation;
        auto obs_tensor = torch::from_blob(current_obs.data(), {(int64_t) current_obs.size()});
        // get global_steps
        pthread_mutex_lock(global_steps_mutex.get());
        global_steps_temp = *global_steps;
        *global_steps += 1;
        pthread_mutex_unlock(global_steps_mutex.get());

        if (global_steps_temp % 1000 == 0) {
            std::cout << "Proceed to step " << global_steps_temp << " at thread " << pthread_self() << std::endl;
        }

        if (global_steps_temp < start_steps) {
            action = std::make_unique<std::vector<float>>(action_space->sample());
        } else {
            // epsilon greedy exploration
            float rand_num = torch::rand({}).item().toFloat();
            if (rand_num > epsilon_greedy) {
                // take agent action
                pthread_mutex_lock(actor_mutex.get()); // shared resource against updater
                auto tensor_action = agent->act_single(obs_tensor).to(torch::kFloat32);
                pthread_mutex_unlock(actor_mutex.get());
                std::vector<float> vector_action(tensor_action.data_ptr<float>(),
                                                 tensor_action.data_ptr<float>() + tensor_action.numel());
                action = std::make_unique<std::vector<float>>(vector_action);
            } else {
                action = std::make_unique<std::vector<float>>(action_space->sample());
            }
        }

        env->step(*action, false, &s);

        // convert data type
        auto action_tensor = torch::from_blob(action->data(), {(int64_t) action->size()});
        auto next_obs_tensor = torch::from_blob(s.observation.data(), {(int64_t) s.observation.size()});
        auto reward_tensor = torch::tensor({s.reward});
        bool true_done = s.done & (!s.timeout);
        auto done_tensor = torch::tensor({true_done}, torch::TensorOptions().dtype(torch::kFloat32));

        pthread_mutex_lock(buffer_mutex.get());
        // store data to the replay buffer
        replay_buffer->add_single({
                                          {"obs",      obs_tensor},
                                          {"act",      action_tensor},
                                          {"next_obs", next_obs_tensor},
                                          {"rew",      reward_tensor},
                                          {"done",     done_tensor}
                                  });
        pthread_mutex_unlock(buffer_mutex.get());

        // determine whether to break
        if (global_steps_temp >= max_global_steps) {
            pthread_mutex_unlock(global_steps_mutex.get());
            break;
        }
    }

    return nullptr;
}

struct LearnerParam {
    std::shared_ptr<PrioritizedReplayBuffer> replay_buffer;
    std::shared_ptr<MlpDQN> agent;
    std::shared_ptr<MlpDQN> actor;
    std::shared_ptr<pthread_mutex_t> global_steps_mutex;
    std::shared_ptr<pthread_mutex_t> buffer_mutex;
    std::vector<std::shared_ptr<pthread_mutex_t>> actor_mutex;
    std::shared_ptr<int64_t> global_steps;
    int64_t max_global_steps;
    int64_t update_after;
    std::shared_ptr<LinearSchedule> beta_scheduler;
    torch::Device device;

    LearnerParam(std::shared_ptr<PrioritizedReplayBuffer> replay_buffer_,
                 std::shared_ptr<MlpDQN> agent_,
                 std::shared_ptr<MlpDQN> actor_,
                 std::shared_ptr<pthread_mutex_t> global_steps_mutex_,
                 std::shared_ptr<pthread_mutex_t> buffer_mutex_,
                 std::vector<std::shared_ptr<pthread_mutex_t>> actor_mutex_,
                 std::shared_ptr<int64_t> global_steps_,
                 int64_t max_global_steps_,
                 int64_t update_after_,
                 std::shared_ptr<LinearSchedule> beta_scheduler_,
                 torch::Device device_) :
            replay_buffer(std::move(replay_buffer_)),
            agent(std::move(agent_)),
            actor(std::move(actor_)),
            global_steps_mutex(std::move(global_steps_mutex_)),
            buffer_mutex(std::move(buffer_mutex_)),
            actor_mutex(std::move(actor_mutex_)),
            global_steps(std::move(global_steps_)),
            max_global_steps(max_global_steps_),
            update_after(update_after_),
            beta_scheduler(std::move(beta_scheduler_)),
            device(device_) {

    }
};


void *learner_fn(void *params) {
    std::cout << "Creating learner thread " << pthread_self() << std::endl;

    auto *learner_param = (LearnerParam *) params;
    auto buffer = learner_param->replay_buffer;
    auto agent = learner_param->agent;
    auto actor = learner_param->agent;
    auto device = learner_param->device;
    auto beta_scheduler = learner_param->beta_scheduler;
    auto global_steps = learner_param->global_steps;
    auto global_steps_mutex = learner_param->global_steps_mutex;
    auto max_global_steps = learner_param->max_global_steps;
    auto actor_mutex = learner_param->actor_mutex;
    auto update_after = learner_param->update_after;
    auto buffer_mutex = learner_param->buffer_mutex;
    int64_t global_steps_temp;

    torch::Device cpu(torch::kCPU);

    while (true) {
        pthread_mutex_lock(global_steps_mutex.get());
        global_steps_temp = *global_steps;
        pthread_mutex_unlock(global_steps_mutex.get());

        if (global_steps_temp >= max_global_steps) {
            break;
        }

        if (global_steps_temp >= update_after) {
            pthread_mutex_lock(buffer_mutex.get());
            auto idx = buffer->generate_idx();
            // get importance weights
            auto weights = buffer->get_weights(*idx, (float) beta_scheduler->value(global_steps_temp));
            // retrieve the actual data
            auto data = *(*buffer)[*idx];
            pthread_mutex_unlock(buffer_mutex.get());
            // training
            auto logs = agent->train_step(data["obs"].to(device),
                                          data["act"].to(device),
                                          data["next_obs"].to(device),
                                          data["rew"].to(device),
                                          data["done"].to(device),
                                          weights->to(device));
            agent->update_target(true);
            // update priority
            pthread_mutex_lock(buffer_mutex.get());
            // TODO: the idx may be override by the newly added items
            buffer->update_priorities(*idx, (*logs)["abs_delta_q"].to(cpu));
            pthread_mutex_unlock(buffer_mutex.get());

            // propagate the weights to actor. The weights of actor are on CPU by default.
            // only need to update the q_network, not the target_q_network
            // try to acquire all the locks
            for (auto &mutex : actor_mutex) {
                pthread_mutex_lock(mutex.get());
            }
            {
                torch::NoGradGuard no_grad;
                for (int i = 0; i < agent->parameters().size(); i++) {
                    auto target_param = actor->parameters().at(i);
                    auto param = agent->parameters().at(i);
                    target_param.data().copy_(param.data().to(cpu));
                }
            }
            for (auto &mutex : actor_mutex) {
                pthread_mutex_unlock(mutex.get());
            }
        }
    }

    return nullptr;
}

static void train_dqn_parallel(
        const std::vector<boost::shared_ptr<Gym::Client>> &clients,
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
        // replay buffer
        float alpha,
        float initial_beta,
        int64_t replay_size,
        // agent parameters
        int64_t mlp_hidden,
        bool double_q,
        float gamma,
        float q_lr,
        float tau,
        float epsilon_greedy,
        // torch
        torch::Device device,
        // parallel
        int64_t num_actors
) {
    // create dummy environment
    std::vector<boost::shared_ptr<Gym::Environment>> envs;
    envs.reserve(num_actors);
    for (int i = 0; i < num_actors; ++i) {
        envs.push_back(clients.at(i)->make(env_id));
    }

    auto env = envs.at(0);
    boost::shared_ptr<Gym::Space> action_space = env->action_space();
    boost::shared_ptr<Gym::Space> observation_space = env->observation_space();
    // create agent
    auto obs_dim = (int64_t) observation_space->box_low.size();
    int64_t act_dim = action_space->discreet_n;
    // setup agent
    auto agent = std::make_shared<MlpDQN>(obs_dim, act_dim, mlp_hidden, double_q, q_lr, gamma, tau);
    auto actor = std::make_shared<MlpDQN>(obs_dim, act_dim, mlp_hidden, double_q, q_lr, gamma, tau);
    agent->to(device);
    // create replay buffer
    ReplayBuffer::str_to_dataspec data_spec = {
            {"obs",      DataSpec({obs_dim}, torch::kFloat32)},
            {"act",      DataSpec({}, torch::kInt64)},
            {"next_obs", DataSpec({obs_dim}, torch::kFloat32)},
            {"rew",      DataSpec({}, torch::kFloat32)},
            {"done",     DataSpec({}, torch::kFloat32)},
    };
    auto buffer = std::make_shared<PrioritizedReplayBuffer>(replay_size, data_spec,
                                                            batch_size, alpha);

    auto beta_scheduler = std::make_shared<LinearSchedule>(epochs * steps_per_epoch, 1.0, initial_beta);
    // create mutex
    std::vector<std::shared_ptr<pthread_mutex_t>> actor_mutex;
    actor_mutex.reserve(num_actors);
    auto global_steps_mutex = std::make_shared<pthread_mutex_t>();
    auto buffer_mutex = std::make_shared<pthread_mutex_t>();
    auto global_steps = std::make_shared<int64_t>(0);
    int64_t max_global_steps = epochs * steps_per_epoch;

    // create actor parameters
    std::vector<std::shared_ptr<ActorParam>> actor_params;
    actor_params.reserve(num_actors);
    for (int i = 0; i < num_actors; ++i) {
        actor_mutex.push_back(std::make_shared<pthread_mutex_t>());
        auto actorParam = std::make_shared<ActorParam>(envs.at(i),
                                                       buffer,
                                                       actor,
                                                       actor_mutex[i],
                                                       buffer_mutex,
                                                       global_steps_mutex,
                                                       global_steps,
                                                       max_global_steps,
                                                       start_steps,
                                                       epsilon_greedy
        );
        actor_params.push_back(actorParam);
    }

    // create learner parameters
    auto learner_param = std::make_shared<LearnerParam>(
            buffer,
            agent,
            actor,
            global_steps_mutex,
            buffer_mutex,
            actor_mutex,
            global_steps,
            max_global_steps,
            update_after,
            beta_scheduler,
            device
    );

    // start the actor thread
    std::vector<pthread_t> actor_threads;
    int ret;
    pthread_t learner_thread;
    for (int i = 0; i < num_actors; ++i) {
        actor_threads.push_back(pthread_t());
        ret = pthread_create(&actor_threads.at(i), nullptr, &actor_fn, (void *) actor_params.at(i).get());
        if (ret != 0) {
            std::cout << "Fail to create actor thread with error code " << ret << std::endl;
        }
    }

    // start the learned thread
    ret = pthread_create(&learner_thread, nullptr, &learner_fn, (void *) learner_param.get());
    if (ret != 0) {
        std::cout << "Fail to create learner thread with error code " << ret << std::endl;
    }

    // wait for all the thread to finish
    pthread_join(learner_thread, nullptr);
    for (int i = 0; i < num_actors; ++i) {
        pthread_join(actor_threads.at(i), nullptr);
    }
}


#endif //HIPC21_DQN_H
