//
// Created by Chi Zhang on 8/5/21.
//

#ifndef HIPC21_TD3_H
#define HIPC21_TD3_H

#include "off_policy_agent.h"
#include "gym/gym.h"
#include "common.h"
#include <cmath>


class TD3Agent : public OffPolicyAgent {
public:
    explicit TD3Agent(const Gym::Space &obs_space,
                      const Gym::Space &act_space,
                      int64_t policy_mlp_hidden = 128,
                      float policy_lr = 3e-4,
                      int64_t q_mlp_hidden = 256,
                      float q_lr = 3e-4,
                      float tau = 5e-3,
                      float gamma = 0.99,
                      float actor_noise = 0.1,
                      float target_noise = 0.2,
                      float noise_clip = 0.5
    ) : actor_noise(actor_noise),
        target_noise(target_noise),
        noise_clip(noise_clip) {

        M_Assert(act_space.type == Gym::Space::BOX, "Only support continuous action space");
        if (act_space.box_shape.size() == 1 && obs_space.box_shape.size() == 1) {
            int64_t obs_dim = obs_space.box_shape[0];
            int64_t act_dim = act_space.box_shape[0];
            this->q_network = std::make_shared<Mlp>(obs_dim, act_dim, q_mlp_hidden);
            this->target_q_network = std::make_shared<Mlp>(obs_dim, act_dim, q_mlp_hidden);
            this->q_optimizer = std::make_unique<torch::optim::Adam>(q_network.ptr()->parameters(),
                                                                     torch::optim::AdamOptions(this->q_lr));
        }

        this->tau = tau;
        this->gamma = gamma;
        this->q_lr = q_lr;


        register_module("q_network", q_network.ptr());
        register_module("target_q_network", target_q_network.ptr());

        update_target(false);
    }

private:
    float actor_noise;
    float target_noise;
    float noise_clip;
};


#endif //HIPC21_TD3_H
