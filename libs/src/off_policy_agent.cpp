//
// Created by Chi Zhang on 8/5/21.
//

#include "agent/off_policy_agent.h"

namespace rlu::agent {

    void OffPolicyAgent::update_target_q(bool soft) {
        if (soft) {
            rlu::functional::soft_update(*target_q_network.ptr(), *q_network.ptr(), tau);
        } else {
            rlu::functional::hard_update(*target_q_network.ptr(), *q_network.ptr());
        }
    }

    void OffPolicyAgent::set_logger(const std::shared_ptr<rlu::logger::EpochLogger> &logger) {
        this->m_logger = logger;
    }

    OffPolicyAgent::OffPolicyAgent(std::shared_ptr<gym::space::Space> obs_space,
                                   std::shared_ptr<gym::space::Space> act_space, float tau, float q_lr, float gamma) :
            obs_space(std::move(obs_space)),
            act_space(std::move(act_space)),
            tau(tau),
            q_lr(q_lr),
            gamma(gamma) {

    }

}