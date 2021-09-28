//
// Created by chi on 9/20/21.
//

#ifndef HIPC21_TESTER_H
#define HIPC21_TESTER_H

#include <agent/off_policy_agent.h>

#include <utility>
#include "gym/gym.h"
#include "logger.h"

namespace rlu::trainer {
    class Tester {
    public:
        explicit Tester(std::shared_ptr<Gym::Environment> env,
                        std::shared_ptr<agent::OffPolicyAgent> test_actor,
                        std::shared_ptr<rlu::logger::EpochLogger> logger,
                        int64_t num_test_episodes,
                        const torch::Device &device);

        void set_env(const std::shared_ptr<Gym::Environment> &env);

        void set_actor(const std::shared_ptr<agent::OffPolicyAgent> &actor);

        void set_logger(const std::shared_ptr<rlu::logger::EpochLogger> &logger);

        void log_tabular();

        void run();

    protected:
        void test_step();


    private:
        std::shared_ptr<rlu::logger::EpochLogger> m_logger;
        int64_t m_num_test_episodes;
        std::shared_ptr<Gym::Environment> m_test_env;
        std::shared_ptr<agent::OffPolicyAgent> m_test_actor;
        torch::Device m_device;
    };
}


#endif //HIPC21_TESTER_H
