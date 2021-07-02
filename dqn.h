//
// Created by chi on 7/1/21.
//

#ifndef HIPC21_DQN_H
#define HIPC21_DQN_H

#include <torch/torch.h>
#include <boost/range/combine.hpp>
#include <utility>
#include "functional.h"

class DQN : public torch::nn::Module {
protected:
    torch::nn::AnyModule q_network;
    torch::nn::AnyModule target_q_network;
    float tau{};
    bool double_q{};
    float q_lr{};
    float gamma{};
    std::shared_ptr<torch::optim::Adam> optimizer;
public:
    void update_target(bool soft) {
        if (soft) {
            soft_update(*target_q_network.ptr(), *q_network.ptr(), tau);
        } else {
            hard_update(*target_q_network.ptr(), *q_network.ptr());
        }
    }

    void train_step(const torch::Tensor &obs,
                    const torch::Tensor &act,
                    const torch::Tensor &next_obs,
                    const torch::Tensor &rew,
                    const torch::Tensor &done) {

        // compute target values
        torch::Tensor target_q_values;
        {
            torch::NoGradGuard no_grad;
            target_q_values = this->target_q_network.forward(next_obs); // shape (None, act_dim)

            if (double_q) {
                auto target_actions = std::get<1>(torch::max(this->q_network.forward(next_obs), -1)); // shape (None,)
                target_q_values = torch::gather(target_q_values, 1, target_actions.unsqueeze(1));
            } else {
                target_q_values = std::get<0>(torch::max(target_q_values, -1));
            }
            target_q_values = rew + gamma * (1. - done) * target_q_values;
        }
        optimizer->zero_grad();
        auto q_values = this->q_network.forward(obs);
        q_values = torch::gather(q_values, 1, act.unsqueeze(1)).squeeze(1);
        auto loss = torch::mse_loss(q_values, target_q_values);
        AT_ASSERT(!std::isnan(loss.template item<float>()));
        loss.backward();
        optimizer->step();
    }

    torch::Tensor act_batch(const torch::Tensor &obs) {
        auto q_values = this->q_network.forward(obs); // shape (None, act_dim)
        auto act = std::get<1>(torch::max(q_values, -1));
        return act;
    }
};


class MlpDQN : public DQN {
private:
    int obs_dim;
    int act_dim;
public:
    MlpDQN(int obs_dim, int act_dim, int mlp_hidden, bool double_q, float q_lr, float gamma) {
        this->q_network = std::make_shared<Mlp>(obs_dim, act_dim, mlp_hidden);
        this->target_q_network = std::make_shared<Mlp>(obs_dim, act_dim, mlp_hidden);
        this->optimizer = std::make_shared<torch::optim::Adam>(q_network.ptr()->parameters(),
                                                               torch::optim::AdamOptions(q_lr));
        this->obs_dim = obs_dim;
        this->act_dim = act_dim;
        this->double_q = double_q;
        this->q_lr = q_lr;
        this->gamma = gamma;

        register_module("q_network", q_network.ptr());
        register_module("target_q_network", target_q_network.ptr());

        update_target(false);
    }
};

class AtariDQN : public DQN {

};


static void train_dqn() {

}

#endif //HIPC21_DQN_H
