//
// Created by chi on 10/13/21.
//

#ifndef HIPC21_DDPG_H
#define HIPC21_DDPG_H

#include "td3.h"

namespace rlu::agent {

    /*
     * DDPG is simply TD3 without ensemble, target noise and policy delay
     */
    class DDPGAgent : public TD3Agent {
    public:
        explicit DDPGAgent(const std::shared_ptr<gym::space::Space> &obs_space,
                           const std::shared_ptr<gym::space::Space> &act_space,
                           int64_t policy_mlp_hidden = 64,
                           float policy_lr = 1e-3,
                           int64_t q_mlp_hidden = 64,
                           float q_lr = 1e-3,
                           float tau = 5e-3,
                           float gamma = 0.99,
                           float actor_noise = 0.1
        );
    };

}

#endif //HIPC21_DDPG_H
