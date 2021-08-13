//
// Created by Chi Zhang on 8/7/21.
//

#ifndef HIPC21_OFF_POLICY_TRAINER_H
#define HIPC21_OFF_POLICY_TRAINER_H


#include <utility>
#include <functional>

#include "agent/off_policy_agent.h"
#include "replay_buffer/replay_buffer.h"
#include "logger.h"
#include "gym/gym.h"
#include "utils/stop_watcher.h"

namespace rlu::trainer {

    class OffPolicyTrainer {
    public:
        explicit OffPolicyTrainer(const std::function<std::shared_ptr<Gym::Environment>()> &env_fn,
                                  const std::function<std::shared_ptr<agent::OffPolicyAgent>()> &agent_fn,
                                  int64_t epochs,
                                  int64_t
                                  steps_per_epoch,
                                  int64_t start_steps,
                                  int64_t
                                  update_after,
                                  int64_t update_every,
                                  int64_t
                                  update_per_step,
                                  int64_t policy_delay,
                                  int64_t
                                  num_test_episodes,
                                  torch::Device device,
                                  int64_t
                                  seed);;

        void setup_logger(std::optional<std::string> exp_name, const std::string &data_dir);

        virtual void setup_replay_buffer(int64_t replay_size, int64_t batch_size);

        virtual void train();

    protected:
        const std::function<std::shared_ptr<Gym::Environment>()> env_fn;
        const std::function<std::shared_ptr<agent::OffPolicyAgent>()> agent_fn;
        std::shared_ptr<rlu::replay_buffer::ReplayBuffer> buffer;
        const std::shared_ptr<Gym::Environment> env;
        const std::shared_ptr<Gym::Environment> test_env;
        std::shared_ptr<rlu::logger::EpochLogger> logger;
        const std::shared_ptr<agent::OffPolicyAgent> agent;
        const int64_t epochs;
        const int64_t steps_per_epoch;
        const int64_t num_test_episodes;
        const int64_t start_steps;
        const int64_t update_after;
        const int64_t update_every;
        const int64_t update_per_step;
        const int64_t policy_delay;
        const torch::Device device;
        const torch::Device cpu = torch::kCPU;

        int64_t seed;
        int64_t total_steps{};
        float episode_rewards{};
        float episode_length{};
        int64_t num_updates{};
        Gym::State s;

    private:
        void test_step();

        void train_step();

        virtual void reset();
    };
}

#endif //HIPC21_OFF_POLICY_TRAINER_H
