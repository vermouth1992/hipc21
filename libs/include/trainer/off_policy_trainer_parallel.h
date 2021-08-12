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
        using OffPolicyTrainer::OffPolicyTrainer;

    public:
        void train() override {
            // start actor threads
            this->start_actor_threads();
            // start learner threads
            this->start_learner_threads();
            // wait and join
            for (auto &thread : actor_threads) {
                pthread_join(thread, nullptr);
            }
            for (auto &thread : learner_threads) {
                pthread_join(thread, nullptr);
            }
        }

    protected:
        virtual void actor_fn_internal() {
            int64_t global_steps_temp;
            Gym::State s;
            s.done = true;
            int64_t max_global_steps = epochs * steps_per_epoch;
            // main loop
            while (true) {
                if (s.done) {
                    env->reset(&s);
                }
                // compute action
                torch::Tensor action;
                // copy observation
                auto current_obs = s.observation;

                pthread_mutex_lock(&global_steps_mutex);
                global_steps_temp = total_steps;
                // create a new thread for testing and logging
                current_global_steps = global_steps_temp;
                this->start_tester_thread();
                // determine whether to break
                if (global_steps_temp >= max_global_steps) {
                    pthread_mutex_unlock(&global_steps_mutex);
                    break;
                }
                total_steps += 1;
                pthread_mutex_unlock(&global_steps_mutex);

                if (global_steps_temp < start_steps) {
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
            }
        }

        virtual void learner_fn_internal() {

        }

        virtual void tester_fn_internal() {
            // copy the weights from learner

        }

        void start_tester_thread() {
            int ret = pthread_create(&tester_thread, nullptr, &tester_fn, this);
            if (ret != 0) {
                MSG("Fail to create tester thread with error code " << ret);
            }
        }

        void start_actor_threads() {
            int ret;
            for (auto &thread : actor_threads) {
                ret = pthread_create(&thread, nullptr, &actor_fn, this);
                if (ret != 0) {
                    MSG("Fail to create actor thread with error code " << ret);
                }
            }
        }

        void start_learner_threads() {
            int ret;
            for (auto &thread : learner_threads) {
                ret = pthread_create(&thread, nullptr, &learner_fn, this);
                if (ret != 0) {
                    MSG("Fail to create learner thread with error code " << ret);
                }
            }
        }

    private:
        static void *actor_fn(void *This) {
            MSG("Running actor thread " << pthread_self());
            ((OffPolicyTrainerParallel *) This)->actor_fn_internal();
            return nullptr;
        }

        static void *learner_fn(void *This) {
            MSG("Running learner thread " << pthread_self());
            ((OffPolicyTrainerParallel *) This)->learner_fn_internal();
            return nullptr;
        }

        static void *tester_fn(void *This) {
            ((OffPolicyTrainerParallel *) This)->tester_fn_internal();
            return nullptr;
        }

        // threads
        std::vector<pthread_t> actor_threads;
        std::vector<pthread_t> learner_threads;
        pthread_t tester_thread;
        // environments
        std::vector<std::shared_ptr<Gym::Environment>> envs;
        // mutexes
        pthread_mutex_t global_steps_mutex;
        // others
        int64_t current_global_steps;

    };
}

#endif //HIPC21_OFF_POLICY_TRAINER_PARALLEL_H
