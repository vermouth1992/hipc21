//
// Created by chi on 7/1/21.
//

#ifndef HIPC21_DQN_H
#define HIPC21_DQN_H

#include <torch/torch.h>
#include <utility>
#include "gym/gym.h"
#include "replay_buffer/replay_buffer_base.h"
#include "common.h"
#include "agent/off_policy_agent.h"
#include "utils/rl_functional.h"
#include "utils/torch_utils.h"
#include "nn/functional.h"
#include "cxxopts.hpp"
#include "fmt/ranges.h"

namespace rlu::agent {

    class DQN : public OffPolicyAgent {
    public:
        explicit DQN(const Gym::Space &obs_space,
                     const Gym::Space &act_space,
                     int64_t mlp_hidden = 128,
                     float q_lr = 1e-3,
                     float gamma = 0.99,
                     float tau = 5e-3,
                     bool double_q = true,
                     float epsilon_greedy = 0.1);

        str_to_tensor train_step(const torch::Tensor &obs,
                                 const torch::Tensor &act,
                                 const torch::Tensor &next_obs,
                                 const torch::Tensor &rew,
                                 const torch::Tensor &done,
                                 const std::optional<torch::Tensor> &importance_weights,
                                 bool update_target) override;

        str_to_tensor_list compute_grad(const torch::Tensor &obs,
                                        const torch::Tensor &act,
                                        const torch::Tensor &next_obs,
                                        const torch::Tensor &rew,
                                        const torch::Tensor &done,
                                        const std::optional<torch::Tensor> &importance_weights,
                                        bool update_target) override;

        void set_grad(const str_to_tensor_list &grads) override;

        void update_step(bool update_target) override;


        torch::Tensor act_single(const torch::Tensor &obs, bool exploration) override;

        torch::Tensor act_test_single(const torch::Tensor &obs);

        void log_tabular() override;

    protected:
        torch::Tensor compute_next_obs_q(const torch::Tensor &next_obs,
                                         const torch::Tensor &rew,
                                         const torch::Tensor &done);


        int64_t m_act_dim;
        bool m_double_q;
        float m_epsilon_greedy;
    };
}

//struct LearnerParam {
//    std::shared_ptr<PrioritizedReplayBuffer> replay_buffer;
//    std::shared_ptr<MlpDQN> agent;
//    std::shared_ptr<MlpDQN> actor;
//    std::shared_ptr<pthread_mutex_t> global_steps_mutex;
//    std::shared_ptr<pthread_mutex_t> buffer_mutex;
//    std::vector<std::shared_ptr<pthread_mutex_t>> actor_mutex;
//    std::shared_ptr<int64_t> global_steps;
//    int64_t max_global_steps;
//    int64_t update_after;
//    std::shared_ptr<LinearSchedule> beta_scheduler;
//    torch::Device device;
//
//    LearnerParam(std::shared_ptr<PrioritizedReplayBuffer> replay_buffer_,
//                 std::shared_ptr<MlpDQN> agent_,
//                 std::shared_ptr<MlpDQN> actor_,
//                 std::shared_ptr<pthread_mutex_t> global_steps_mutex_,
//                 std::shared_ptr<pthread_mutex_t> buffer_mutex_,
//                 std::vector<std::shared_ptr<pthread_mutex_t>> actor_mutex_,
//                 std::shared_ptr<int64_t> global_steps_,
//                 int64_t max_global_steps_,
//                 int64_t update_after_,
//                 std::shared_ptr<LinearSchedule> beta_scheduler_,
//                 torch::Device device_) :
//            replay_buffer(std::move(replay_buffer_)),
//            agent(std::move(agent_)),
//            actor(std::move(actor_)),
//            global_steps_mutex(std::move(global_steps_mutex_)),
//            buffer_mutex(std::move(buffer_mutex_)),
//            actor_mutex(std::move(actor_mutex_)),
//            global_steps(std::move(global_steps_)),
//            max_global_steps(max_global_steps_),
//            update_after(update_after_),
//            beta_scheduler(std::move(beta_scheduler_)),
//            device(device_) {
//
//    }
//};
//
//
//void *learner_fn(void *params) {
//    std::cout << "Creating learner thread " << pthread_self() << std::endl;
//
//    auto *learner_param = (LearnerParam *) params;
//    auto buffer = learner_param->replay_buffer;
//    auto agent = learner_param->agent;
//    auto actor = learner_param->agent;
//    auto device = learner_param->device;
//    auto beta_scheduler = learner_param->beta_scheduler;
//    auto global_steps = learner_param->global_steps;
//    auto global_steps_mutex = learner_param->global_steps_mutex;
//    auto max_global_steps = learner_param->max_global_steps;
//    auto actor_mutex = learner_param->actor_mutex;
//    auto update_after = learner_param->update_after;
//    auto buffer_mutex = learner_param->buffer_mutex;
//    int64_t global_steps_temp;
//
//    torch::Device cpu(torch::kCPU);
//
//    while (true) {
//        pthread_mutex_lock(global_steps_mutex.get());
//        global_steps_temp = *global_steps;
//        pthread_mutex_unlock(global_steps_mutex.get());
//
//        if (global_steps_temp >= max_global_steps) {
//            break;
//        }
//
//        if (global_steps_temp >= update_after) {
//            pthread_mutex_lock(buffer_mutex.get());
//            auto idx = buffer->generate_idx();
//            // get importance weights
//            auto weights = buffer->get_weights(*idx, (float) beta_scheduler->value(global_steps_temp));
//            // retrieve the actual data
//            auto data = *(*buffer)[*idx];
//            pthread_mutex_unlock(buffer_mutex.get());
//            // training
//            auto logs = agent->train_step(data["obs"].to(device),
//                                          data["act"].to(device),
//                                          data["next_obs"].to(device),
//                                          data["rew"].to(device),
//                                          data["done"].to(device),
//                                          weights->to(device));
//            agent->update_target_q(true);
//            // update priority
//            pthread_mutex_lock(buffer_mutex.get());
//            // TODO: the idx may be override by the newly added items
//            buffer->update_priorities(*idx, logs["abs_delta_q"].to(cpu));
//            pthread_mutex_unlock(buffer_mutex.get());
//
//            // propagate the weights to actor. The weights of actor are on CPU by default.
//            // only need to update the q_network, not the target_q_network
//            // try to acquire all the locks
//            for (auto &mutex : actor_mutex) {
//                pthread_mutex_lock(mutex.get());
//            }
//            {
//                torch::NoGradGuard no_grad;
//                for (unsigned long i = 0; i < agent->parameters().size(); i++) {
//                    auto target_param = actor->parameters().at(i);
//                    auto param = agent->parameters().at(i);
//                    target_param.data().copy_(param.data().to(cpu));
//                }
//            }
//            for (auto &mutex : actor_mutex) {
//                pthread_mutex_unlock(mutex.get());
//            }
//        }
//    }
//
//    return nullptr;
//}
//
//static void train_dqn_parallel(
//        const std::vector<std::shared_ptr<Gym::Client>> &clients,
//        const std::string &env_id,
//        int64_t epochs,
//        int64_t steps_per_epoch,
//        int64_t start_steps,
//        int64_t update_after,
//        int64_t update_every,
//        int64_t update_per_step,
//        int64_t policy_delay,
//        int64_t batch_size,
//        int64_t num_test_episodes,
//        int64_t seed,
//        // replay buffer
//        float alpha,
//        float initial_beta,
//        int64_t replay_size,
//        // agent parameters
//        int64_t mlp_hidden,
//        bool double_q,
//        float gamma,
//        float q_lr,
//        float tau,
//        float epsilon_greedy,
//        // torch
//        torch::Device device,
//        // parallel
//        int64_t num_actors
//) {
//    // create dummy environment
//    std::vector<std::shared_ptr<Gym::Environment>> envs;
//    envs.reserve(num_actors);
//    for (int i = 0; i < num_actors; ++i) {
//        envs.push_back(clients.at(i)->make(env_id));
//    }
//
//    auto env = envs.at(0);
//    std::shared_ptr<Gym::Space> action_space = env->action_space();
//    std::shared_ptr<Gym::Space> observation_space = env->observation_space();
//    // create agent
//    auto obs_dim = (int64_t) observation_space->box_low.size();
//    int64_t act_dim = action_space->discreet_n;
//    // setup agent
//    auto agent = std::make_shared<MlpDQN>(obs_dim, act_dim, mlp_hidden, double_q, q_lr, gamma, tau, epsilon_greedy);
//    auto actor = std::make_shared<MlpDQN>(obs_dim, act_dim, mlp_hidden, double_q, q_lr, gamma, tau, epsilon_greedy);
//    agent->to(device);
//    // create replay buffer
//    ReplayBuffer::str_to_dataspec data_spec = {
//            {"obs",      DataSpec({obs_dim}, torch::kFloat32)},
//            {"act",      DataSpec({}, torch::kInt64)},
//            {"next_obs", DataSpec({obs_dim}, torch::kFloat32)},
//            {"rew",      DataSpec({}, torch::kFloat32)},
//            {"done",     DataSpec({}, torch::kFloat32)},
//    };
//    auto buffer = std::make_shared<PrioritizedReplayBuffer>(replay_size, data_spec,
//                                                            batch_size, alpha);
//
//    auto beta_scheduler = std::make_shared<LinearSchedule>(epochs * steps_per_epoch, 1.0, initial_beta);
//    // create mutex
//    std::vector<std::shared_ptr<pthread_mutex_t>> actor_mutex;
//    actor_mutex.reserve(num_actors);
//    auto global_steps_mutex = std::make_shared<pthread_mutex_t>();
//    auto buffer_mutex = std::make_shared<pthread_mutex_t>();
//    auto global_steps = std::make_shared<int64_t>(0);
//    int64_t max_global_steps = epochs * steps_per_epoch;
//
//    // create actor parameters
//    std::vector<std::shared_ptr<ActorParam>> actor_params;
//    actor_params.reserve(num_actors);
//    for (int i = 0; i < num_actors; ++i) {
//        actor_mutex.push_back(std::make_shared<pthread_mutex_t>());
//        auto actorParam = std::make_shared<ActorParam>(envs.at(i),
//                                                       buffer,
//                                                       actor,
//                                                       actor_mutex[i],
//                                                       buffer_mutex,
//                                                       global_steps_mutex,
//                                                       global_steps,
//                                                       max_global_steps,
//                                                       start_steps,
//                                                       epsilon_greedy
//        );
//        actor_params.push_back(actorParam);
//    }
//
//    // create learner parameters
//    auto learner_param = std::make_shared<LearnerParam>(
//            buffer,
//            agent,
//            actor,
//            global_steps_mutex,
//            buffer_mutex,
//            actor_mutex,
//            global_steps,
//            max_global_steps,
//            update_after,
//            beta_scheduler,
//            device
//    );
//
//}


#endif //HIPC21_DQN_H
