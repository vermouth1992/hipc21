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
                                          int64_t num_learners);

        virtual ~OffPolicyTrainerParallel();

        [[nodiscard]] size_t get_num_learners() const;

        void setup_environment() override;

        void train() override;

    protected:
        /*
         * atomically get the global environmental steps with optional increments
         */
        int64_t get_global_steps(bool increment);

        virtual void actor_fn_internal(size_t index);

        virtual void learner_fn_internal(size_t index);

        virtual void tester_fn_internal(int64_t epoch);

        void start_tester_thread(int64_t epoch);

        void start_actor_threads();

        void start_learner_threads();

        static void *actor_fn(void *param_);

        static void *learner_fn(void *param_);

        static void *tester_fn(void *param_);

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
        /*
         * Aggregate the gradients
         */
        str_to_tensor_list aggregate_grads();

    };
}

#endif //HIPC21_OFF_POLICY_TRAINER_PARALLEL_H
