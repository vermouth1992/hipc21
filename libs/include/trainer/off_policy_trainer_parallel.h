//
// Created by Chi Zhang on 8/7/21.
//

#ifndef HIPC21_OFF_POLICY_TRAINER_PARALLEL_H
#define HIPC21_OFF_POLICY_TRAINER_PARALLEL_H

#include "off_policy_trainer.h"
#include "common.h"
#include <pthread.h>
#include <vector>

namespace rlu::trainer {

    class OffPolicyTrainerParallel : public OffPolicyTrainer {
    public:
        explicit OffPolicyTrainerParallel(const std::function<std::shared_ptr<Gym::Environment>()> &env_fn,
                                          const std::function<std::shared_ptr<agent::OffPolicyAgent>()> &agent_fn,
                                          int64_t epochs,
                                          int64_t steps_per_epoch,
                                          int64_t start_steps,
                                          int64_t update_after,
                                          int64_t update_every,
                                          int64_t update_per_step,
                                          int64_t policy_delay,
                                          int64_t num_test_episodes,
                                          torch::Device device,
                                          int64_t seed,
                                          int64_t num_actors,
                                          int64_t num_learners) :
                OffPolicyTrainer(
                        env_fn,
                        agent_fn,
                        epochs,
                        steps_per_epoch,
                        start_steps,
                        update_after,
                        update_every,
                        update_per_step,
                        policy_delay,
                        num_test_episodes,
                        device,
                        seed
                ),
                actor(agent_fn()) {
            for (int i = 0; i < num_actors; i++) {
                envs.push_back(env_fn());
                actor_mutexes.emplace_back();
                actor_threads.emplace_back();
            }
            for (int i = 0; i < num_learners; i++) {
                learner_threads.emplace_back();
                grads.push_back(nullptr);
            }
            num_finished_learners = 0;
            rlu::functional::hard_update(*actor, *agent);
        }

        [[nodiscard]] size_t get_num_learners() const {
            return learner_threads.size();
        }

        void train() override {
            // initialize
            torch::manual_seed(seed);
            watcher.start();
            // start actor threads
            this->start_actor_threads();
            // start learner threads
            this->start_learner_threads();
            // wait and join
            for (auto &thread: actor_threads) {
                pthread_join(thread, nullptr);
            }
            for (auto &thread: learner_threads) {
                pthread_join(thread, nullptr);
            }
        }

    protected:
        /*
         * atomically get the global environmental steps with optional increments
         */
        int64_t get_global_steps(bool increment) {
            int64_t global_steps_temp;
            pthread_mutex_lock(&global_steps_mutex);
            global_steps_temp = total_steps;
            if (increment) {
                total_steps += 1;
            }
            pthread_mutex_unlock(&global_steps_mutex);
            return global_steps_temp;
        }

        virtual void actor_fn_internal(size_t index) {
            // get environment
            auto curr_env = envs.at(index);

            Gym::State s;
            curr_env->reset(&s);
            int64_t max_global_steps = epochs * steps_per_epoch;
            // logging
            float episode_rewards = 0;
            float episode_length = 0;
            // main loop
            while (true) {
                // compute action
                torch::Tensor action;
                // copy observation
                auto current_obs = s.observation;

                int64_t global_steps_temp = this->get_global_steps(true);
                // create a new thread for testing and logging
                if (global_steps_temp % steps_per_epoch == 0) {
                    // create a new agent. Copy the weights from the current learner
                    std::shared_ptr<agent::OffPolicyAgent> test_actor = agent_fn();
                    pthread_mutex_lock(&test_actor_mutex);
                    rlu::functional::hard_update(*test_actor, *actor);
                    pthread_mutex_unlock(&test_actor_mutex);

                    this->start_tester_thread(test_actor,
                                              {
                                                      {"epoch", global_steps_temp / steps_per_epoch}
                                              });
                }
                // determine whether to break
                if (global_steps_temp >= max_global_steps) {
                    break;
                }


                if (global_steps_temp < start_steps) {
                    action = curr_env->action_space()->sample();
                } else {
                    // hold actor mutex. Should be mutually exclusive when copying the weights from learner to the actor
                    pthread_mutex_lock(&actor_mutexes[index]);
                    action = actor->act_single(current_obs.to(device), true).to(cpu);
                    pthread_mutex_unlock(&actor_mutexes[index]);
                }

                // environment step
                curr_env->step(action, false, &s);

                // TODO: need to see if it is true done or done due to reaching the maximum length.
                // convert data type
                auto reward_tensor = torch::tensor({s.reward});
                bool true_done = s.done & (!s.timeout);
                auto done_tensor = torch::tensor({true_done},
                                                 torch::TensorOptions().dtype(torch::kFloat32));

                // store the data in a temporary buffer and wait for every batch size to store together
                // step 1: secure an index. If can't, wait in the conditional variable

                // step 2: add data to the index without lock

                // step 3: if the temporary buffer is full, compute priority and add to the replay buffer

                // compute the priority

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

                    curr_env->reset(&s);
                    episode_rewards = 0.;
                    episode_length = 0;
                }
            }
        }

        virtual void learner_fn_internal(size_t index) {
            int64_t max_global_steps = epochs * steps_per_epoch;
            while (true) {
                // get global steps
                int64_t global_steps_temp = this->get_global_steps(false);
                if (global_steps_temp >= max_global_steps) {
                    break;
                }
                bool update_target = num_updates % policy_delay == 0;
                // sample a batch of data.
                // generate index
                auto idx = buffer->generate_idx();
                // retrieve the actual data
                auto data = buffer->operator[](idx);
                // compute the gradients
                auto local_grads = agent->compute_grad(data["obs"].to(device),
                                                       data["act"].to(device),
                                                       data["next_obs"].to(device),
                                                       data["rew"].to(device),
                                                       data["done"].to(device),
                                                       std::nullopt,
                                                       update_target);
                // store the local grad in the shared memory
                grads.at(index) = std::make_shared<str_to_tensor_list>(local_grads);

                // atomically increase the done flag
                pthread_mutex_lock(&learner_barrier);
                num_finished_learners += 1;
                if (num_finished_learners == this->get_num_learners()) {
                    // last thread aggregate the gradients, otherwise, conditional wait for the done signal
                    auto aggregated_grads = this->aggregate_grads();
                    // set the gradients
                    agent->set_grad(aggregated_grads);
                    // optimizer step
                    agent->update_step(update_target);
                    num_updates += 1;
                    // atomically copy weights to actors and test_actors
                    for (auto &m: actor_mutexes) {
                        pthread_mutex_lock(&m);
                    }
                    // TODO: need to optimize for PCIe transfer. First transfer to a CPU agent.
                    // Then, perform atomic weight copy
                    pthread_mutex_lock(&test_actor_mutex);
                    rlu::functional::hard_update(*actor, *agent);
                    pthread_mutex_unlock(&test_actor_mutex);

                    for (auto &m: actor_mutexes) {
                        pthread_mutex_unlock(&m);
                    }
                    num_finished_learners = 0;
                    // broadcast
                    pthread_cond_broadcast(&learner_cond);
                } else {
                    // wait the aggregator
                    pthread_cond_wait(&learner_cond, &learner_barrier);
                }

                pthread_mutex_unlock(&learner_barrier);
            }
        }

        virtual void tester_fn_internal(const std::shared_ptr<agent::OffPolicyAgent> &test_actor,
                                        const std::unordered_map<std::string, float> &param_) {
            // perform logging
            logger->log_tabular("Epoch", param_.at("epoch"));
            logger->log_tabular("EpRet", std::nullopt, true);
            logger->log_tabular("EpLen", std::nullopt, false, true);
            logger->log_tabular("TotalEnvInteracts", (float) total_steps);
            agent->log_tabular();

            // test the current policy
            for (int i = 0; i < num_test_episodes; ++i) {
                test_step(test_actor);
            }
            // logging
            watcher.lap();

            logger->log_tabular("TestEpRet", std::nullopt, true);
            logger->log_tabular("TestEpLen", std::nullopt, false, true);
            logger->log_tabular("Time", (float) watcher.seconds());

            logger->dump_tabular();
        }

        void start_tester_thread(const std::shared_ptr<agent::OffPolicyAgent> &test_actor,
                                 const std::unordered_map<std::string, float> &param_) {
            auto param = std::make_shared<std::tuple<OffPolicyTrainerParallel *, std::unordered_map<std::string, float>,
                    std::shared_ptr<agent::OffPolicyAgent>>>(this, param_, test_actor);
            int ret = pthread_create(&tester_thread, nullptr, &tester_fn, param.get());
            if (ret != 0) {
                MSG("Fail to create tester thread with error code " << ret);
            }
        }

        void start_actor_threads() {
            int ret;
            for (size_t i = 0; i < actor_threads.size(); i++) {
                auto param = std::make_shared<std::pair<OffPolicyTrainerParallel *, size_t>>(this, i);
                ret = pthread_create(&actor_threads[i], nullptr, &actor_fn, param.get());
                if (ret != 0) {
                    MSG("Fail to create actor thread with error code " << ret);
                }
            }
        }

        void start_learner_threads() {
            int ret;
            for (size_t i = 0; i < learner_threads.size(); i++) {
                auto param = std::make_shared<std::pair<OffPolicyTrainerParallel *, size_t>>(this, i);
                ret = pthread_create(&learner_threads[i], nullptr, &learner_fn, param.get());
                if (ret != 0) {
                    MSG("Fail to create learner thread with error code " << ret);
                }
            }
        }

        // agents
        const std::shared_ptr<agent::OffPolicyAgent> actor;
        // threads
        std::vector<pthread_t> actor_threads;
        std::vector<pthread_t> learner_threads;
        pthread_t tester_thread{};
        // environments
        std::vector<std::shared_ptr<Gym::Environment>> envs;
        // mutexes
        pthread_mutex_t global_steps_mutex{};
        std::vector<pthread_mutex_t> actor_mutexes;
        pthread_mutex_t test_actor_mutex{};
        // gradients
        std::vector<std::shared_ptr<str_to_tensor_list>> grads;
        // others
        size_t num_finished_learners;
        pthread_mutex_t learner_barrier{};
        pthread_cond_t learner_cond{};

    private:
        static void *actor_fn(void *param_) {
            auto param = (std::pair<OffPolicyTrainerParallel *, size_t> *) param_;
            OffPolicyTrainerParallel *This = param->first;
            size_t index = param->second;
            MSG("Running actor thread " << pthread_self());
            This->actor_fn_internal(index);
            return nullptr;
        }

        static void *learner_fn(void *param_) {
            auto param = (std::pair<OffPolicyTrainerParallel *, size_t> *) param_;
            OffPolicyTrainerParallel *This = param->first;
            size_t index = param->second;
            MSG("Running learner thread " << pthread_self());
            This->learner_fn_internal(index);
            return nullptr;
        }

        static void *tester_fn(void *param_) {
            auto param = (std::tuple<OffPolicyTrainerParallel *, std::unordered_map<std::string, float>, std::shared_ptr<agent::OffPolicyAgent> > *) param_;
            OffPolicyTrainerParallel *This = std::get<0>(*param);
            ((OffPolicyTrainerParallel *) This)->tester_fn_internal(std::get<2>(*param),
                                                                    std::get<1>(*param));
            return nullptr;
        }

        /*
         * Aggregate the gradients
         */
        str_to_tensor_list aggregate_grads() {
            str_to_tensor_list result;
            for (auto &it: *grads.at(0)) {
                result[it.first] = torch::autograd::variable_list(it.second.size());
                // for each parameter
                for (size_t i = 0; i < it.second.size(); i++) {
                    std::vector<torch::Tensor> current_grad;
                    // for each learner
                    for (auto &learner_grad: grads) {
                        auto param_grad = learner_grad->at(it.first); // a list of tensor
                        current_grad.push_back(param_grad.at(i));
                    }
                    // aggregate
                    result[it.first].at(i) = torch::mean(torch::stack(current_grad, 0), 0);
                }
            }
            return result;
        }

    };
}

#endif //HIPC21_OFF_POLICY_TRAINER_PARALLEL_H
