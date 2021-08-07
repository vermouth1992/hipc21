//
// Created by Chi Zhang on 8/5/21.
//

#include "off_policy_agent.h"

void OffPolicyAgent::update_target_q(bool soft) {
    if (soft) {
        soft_update(*target_q_network.ptr(), *q_network.ptr(), tau);
    } else {
        hard_update(*target_q_network.ptr(), *q_network.ptr());
    }
}

void OffPolicyAgent::set_logger(const std::shared_ptr<EpochLogger> &logger) {
    this->m_logger = logger;
}

OffPolicyAgent::OffPolicyAgent(float tau, float q_lr, float gamma) : tau(tau), q_lr(q_lr), gamma(gamma) {

}
