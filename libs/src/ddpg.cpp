//
// Created by chi on 10/13/21.
//

#include "agent/ddpg.h"

rlu::agent::DDPGAgent::DDPGAgent(const std::shared_ptr<gym::space::Space> &obs_space,
                                 const std::shared_ptr<gym::space::Space> &act_space,
                                 int64_t policy_mlp_hidden,
                                 float policy_lr, int64_t q_mlp_hidden, float q_lr, float tau, float gamma,
                                 float actor_noise) :
        TD3Agent(obs_space, act_space, policy_mlp_hidden,
                 policy_lr, q_mlp_hidden, 1, q_lr, tau, gamma,
                 actor_noise, 0.0, 0.0) {

}
