//
// Created by chi on 9/20/21.
//

#include "trainer/tester.h"

rlu::trainer::Tester::Tester(std::shared_ptr<Gym::Environment> env, std::shared_ptr<agent::OffPolicyAgent> test_actor,
                             std::shared_ptr<rlu::logger::EpochLogger> logger, int64_t num_test_episodes,
                             const torch::Device &device) :
        m_logger(std::move(logger)),
        m_num_test_episodes(num_test_episodes),
        m_test_env(std::move(env)),
        m_test_actor(std::move(test_actor)),
        m_device(device) {

}

void rlu::trainer::Tester::log_tabular() {
    m_logger->log_tabular("TestEpRet", std::nullopt, true);
    m_logger->log_tabular("TestEpLen", std::nullopt, false, true);
}

void rlu::trainer::Tester::run() {
    spdlog::debug("Start testing");
    for (int i = 0; i < m_num_test_episodes; ++i) {
        test_step();
    }
    spdlog::debug("Finish testing");
}

void rlu::trainer::Tester::test_step() {
    Gym::State test_s;
    // testing variables
    m_test_env->reset(&test_s);
    float test_episode_reward = 0;
    float test_episode_length = 0;
    while (true) {
        auto tensor_action = m_test_actor->act_single(test_s.observation.to(m_device), false).to(torch::kCPU);
        m_test_env->step(tensor_action, false, &test_s);
        test_episode_reward += test_s.reward;
        test_episode_length += 1;
        if (test_s.done) break;
    }
    m_logger->store("TestEpRet", test_episode_reward);
    m_logger->store("TestEpLen", test_episode_length);
}

void rlu::trainer::Tester::set_env(const std::shared_ptr<Gym::Environment> &env) {
    this->m_test_env = env;
}

void rlu::trainer::Tester::set_actor(const std::shared_ptr<agent::OffPolicyAgent> &actor) {
    this->m_test_actor = actor;
}

void rlu::trainer::Tester::set_logger(const std::shared_ptr<rlu::logger::EpochLogger> &logger) {
    this->m_logger = logger;
}
