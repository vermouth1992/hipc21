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

        ~OffPolicyTrainerParallel() override;

        [[nodiscard]] size_t get_num_learners() const;

        [[nodiscard]] size_t get_num_actors() const;

        void setup_environment() override;

        void setup_replay_buffer(int64_t replay_size, int64_t batch_size) override;

        void train() override;

    protected:
        /*
         * atomically get the global environmental steps with optional increments
         */
        int64_t get_global_steps(bool increment);

        int64_t get_update_steps(bool increment);

        void actor_wait_for_learner(int64_t global_steps_temp);

        void wake_up_actor();

        void wake_up_learner();

        int64_t get_actor_index();

        int64_t get_learner_index();

        void learner_wait_to_start();

        void log(int64_t global_steps_temp, int64_t num_updates_temp);

        virtual void actor_fn_internal();

        virtual void learner_fn_internal();

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
        pthread_mutex_t update_steps_mutex{};
        pthread_cond_t update_steps_cond{};
        std::vector<pthread_mutex_t> actor_mutexes;
        pthread_mutex_t test_actor_mutex{};
        std::vector<pthread_mutex_t> temp_buffer_mutex;
        pthread_mutex_t buffer_mutex{};
        // gradients
        std::vector<std::shared_ptr<str_to_tensor_list>> grads;
        // others
        size_t num_finished_learners;
        pthread_mutex_t learner_barrier{};
        pthread_cond_t learner_cond{};
        int64_t current_actor_index;
        int64_t current_learning_index;
        pthread_mutex_t actor_index_mutex{};
        pthread_mutex_t learning_index_mutex{};
        std::vector<pthread_mutex_t> agent_mutexes;

    private:
        /*
         * Aggregate the gradients
         */
        str_to_tensor_list aggregate_grads();

    };
}

#endif //HIPC21_OFF_POLICY_TRAINER_PARALLEL_H
