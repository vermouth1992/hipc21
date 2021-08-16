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
                ) {
            for (int i = 0; i < num_actors; i++) {
                envs.push_back(env_fn());
                actor_mutexes.emplace_back();
                actor_threads.emplace_back();
            }
            for (int i = 0; i < num_learners; i++) {
                learner_threads.emplace_back();
            }
        }

    public:
        void train() override {
            // start actor threads
            this->start_actor_threads();
            // start learner threads
            this->start_learner_threads();
            // start aggregator thread
            this->start_aggregator_thread();
            // wait and join
            for (auto &thread : actor_threads) {
                pthread_join(thread, nullptr);
            }
            for (auto &thread : learner_threads) {
                pthread_join(thread, nullptr);
            }
        }

    protected:
        virtual void actor_fn_internal(size_t index) {
            // get environment
            auto curr_env = envs.at(index);
            int64_t global_steps_temp;
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

                pthread_mutex_lock(&global_steps_mutex);
                global_steps_temp = total_steps;
                // create a new thread for testing and logging
                if (global_steps_temp % steps_per_epoch == 0) {
                    current_global_steps = global_steps_temp;
                    this->start_tester_thread();
                }
                // determine whether to break
                if (global_steps_temp >= max_global_steps) {
                    pthread_mutex_unlock(&global_steps_mutex);
                    break;
                }
                total_steps += 1;
                pthread_mutex_unlock(&global_steps_mutex);

                if (global_steps_temp < start_steps) {
                    action = curr_env->action_space()->sample();
                } else {
                    // hold actor mutex. Should be mutually exclusive when copying the weights from learner to the actor
                    pthread_mutex_lock(&actor_mutexes[index]);
                    action = agent->act_single(current_obs.to(device), true).to(cpu);
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
            // sample a batch of data.

            // compute the gradients

            // atomically increase the done flag

            // conditional wait for aggregator to update the weights
        }

        virtual void aggregator_fn_internal() {
            // conditional wait for the done flag

            // aggregate the gradients

            // set the gradients

            // optimizer step

            // set done flag

            // atomically copy weights to actors
        }

        virtual void tester_fn_internal() {
            // copy the weights from learner. Should be mutually exclusive with optimizer step.

            // run tester function
        }

        void start_tester_thread() {
            int ret = pthread_create(&tester_thread, nullptr, &tester_fn, this);
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

        void start_aggregator_thread() {
            int ret = pthread_create(&aggregator_thread, nullptr, &aggregator_fn, this);
            if (ret != 0) {
                MSG("Fail to create tester thread with error code " << ret);
            }
        }

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

        static void *tester_fn(void *This) {
            ((OffPolicyTrainerParallel *) This)->tester_fn_internal();
            return nullptr;
        }

        static void *aggregator_fn(void *This) {
            ((OffPolicyTrainerParallel *) This)->aggregator_fn_internal();
            return nullptr;
        }
        // agents

        // threads
        std::vector<pthread_t> actor_threads;
        std::vector<pthread_t> learner_threads;
        pthread_t tester_thread{};
        pthread_t aggregator_thread{};
        // environments
        std::vector<std::shared_ptr<Gym::Environment>> envs;
        // mutexes
        pthread_mutex_t global_steps_mutex{};
        std::vector<pthread_mutex_t> actor_mutexes;
        // others
        int64_t current_global_steps{};

    };
}

#endif //HIPC21_OFF_POLICY_TRAINER_PARALLEL_H
