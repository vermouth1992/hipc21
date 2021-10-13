//
// Created by Chi Zhang on 8/5/21.
//

#include "agent/td3.h"

namespace rlu::agent {

    TD3Agent::TD3Agent(const Gym::Space &obs_space,
                       const Gym::Space &act_space,
                       int64_t policy_mlp_hidden,
                       float policy_lr,
                       int64_t q_mlp_hidden,
                       int64_t num_q_ensembles,
                       float q_lr, float tau, float gamma, float actor_noise, float target_noise,
                       float noise_clip) : OffPolicyAgent(tau, q_lr, gamma),
                                           actor_noise(actor_noise),
                                           target_noise(target_noise),
                                           noise_clip(noise_clip),
                                           act_lim(act_space.box_high.index({0}).item<float>()) {

        M_Assert(act_space.type == Gym::Space::BOX, "Only support continuous action space");
        if (act_space.box_shape.size() == 1 && obs_space.box_shape.size() == 1) {
            int64_t obs_dim = obs_space.box_shape[0];
            int64_t act_dim = act_space.box_shape[0];
            this->q_network = register_module("q_network",
                                              std::make_shared<rlu::nn::EnsembleMinQNet>(obs_dim, act_dim,
                                                                                         q_mlp_hidden,
                                                                                         num_q_ensembles));
            this->target_q_network = register_module("target_q_network",
                                                     std::make_shared<rlu::nn::EnsembleMinQNet>(obs_dim, act_dim,
                                                                                                q_mlp_hidden,
                                                                                                num_q_ensembles));
            this->q_optimizer = std::make_unique<torch::optim::Adam>(q_network.ptr()->parameters(),
                                                                     torch::optim::AdamOptions(q_lr));
            this->policy_net = register_module("policy_network", rlu::nn::build_mlp(obs_dim, act_dim,
                                                                                    policy_mlp_hidden, 3, "relu", false,
                                                                                    "tanh"));
            this->target_policy_net = register_module("target_policy_network",
                                                      rlu::nn::build_mlp(obs_dim, act_dim,
                                                                         policy_mlp_hidden, 3, "relu", false,
                                                                         "tanh"));
            this->policy_optimizer = std::make_unique<torch::optim::Adam>(policy_net.ptr()->parameters(),
                                                                          torch::optim::AdamOptions(policy_lr));

        } else {
            throw std::runtime_error("Unsupported environment");
        }

        this->tau = tau;
        this->gamma = gamma;
        this->q_lr = q_lr;

        this->update_target_q(false);
        this->update_target_policy(false);
    }

    void TD3Agent::update_target_policy(bool soft) {
        if (soft) {
            rlu::functional::soft_update(*target_policy_net.ptr(), *policy_net.ptr(), tau);
        } else {
            rlu::functional::hard_update(*target_policy_net.ptr(), *policy_net.ptr());
        }
    }

    void TD3Agent::log_tabular() {
        for (int i = 0; i < 2; i++) {
            m_logger->log_tabular(fmt::format("Q{}Vals", i + 1), std::nullopt, true);
        }
        m_logger->log_tabular("LossPi", std::nullopt, false, true);
        m_logger->log_tabular("LossQ", std::nullopt, false, true);
    }

    str_to_tensor
    TD3Agent::train_step(const torch::Tensor &obs, const torch::Tensor &act, const torch::Tensor &next_obs,
                         const torch::Tensor &rew, const torch::Tensor &done,
                         const std::optional<torch::Tensor> &importance_weights, bool update_target) {
        auto info = this->update_q_net(obs, act, next_obs, rew, done, importance_weights);
        if (update_target) {
            this->update_actor(obs);
            this->update_target_q(true);
            this->update_target_policy(true);
        }
        return info;
    }

    torch::Tensor TD3Agent::act_single(const torch::Tensor &obs, bool exploration) {
        torch::NoGradGuard no_grad;
        auto obs_batch = obs.unsqueeze(0);
        auto pi_final = this->policy_net.forward(obs_batch); // (1, act_dim)
        pi_final = pi_final.index({0});
        if (exploration) {
            auto noise = torch::randn_like(pi_final) * this->actor_noise;
            pi_final = pi_final + noise;
            pi_final = torch::clip(pi_final, -this->act_lim, this->act_lim);
        }
        return pi_final;
    }

    torch::Tensor
    TD3Agent::compute_target_q(const torch::Tensor &next_obs, const torch::Tensor &rew, const torch::Tensor &done) {
        torch::NoGradGuard no_grad;
        torch::Tensor next_action = this->target_policy_net.forward(next_obs);
        if (this->target_noise > 0) {
            auto epsilon = torch::randn(next_action.sizes()) * this->target_noise;
            epsilon = torch::clip(epsilon, -this->noise_clip, this->noise_clip);
            next_action = next_action + epsilon;
            next_action = torch::clip(next_action, -this->act_lim, this->act_lim);
        }
        auto next_q_value = this->target_q_network.forward(next_obs, next_action, true);
        return rew + gamma * (1. - done) * next_q_value;
    }

    str_to_tensor
    TD3Agent::update_q_net(const torch::Tensor &obs, const torch::Tensor &act, const torch::Tensor &next_obs,
                           const torch::Tensor &rew, const torch::Tensor &done,
                           const std::optional<torch::Tensor> &importance_weights) {
        auto q_target = this->compute_target_q(next_obs, rew, done);
        q_optimizer->zero_grad();
        auto q_values = this->q_network.forward(obs, act, false); // (num_ensemble, None)
        auto q_values_loss = 0.5 * torch::square(torch::unsqueeze(q_target, 0) - q_values);
        q_values_loss = torch::sum(q_values_loss, 0);  // (None,)
        // apply importance weights
        if (importance_weights != std::nullopt) {
            q_values_loss = q_values_loss * importance_weights.value();
        }
        q_values_loss = torch::mean(q_values_loss);
        q_values_loss.backward();
        q_optimizer->step();
        for (int i = 0; i < 2; i++) {
            m_logger->store(fmt::format("Q{}Vals", i + 1),
                            rlu::nn::convert_tensor_to_flat_vector<float>(q_values.index({0}).detach()));
        }
        m_logger->store("LossQ", q_values_loss.item<float>());

        auto priority = torch::abs(std::get<0>(torch::min(q_values, 0)) - q_target).detach().cpu();
        str_to_tensor info{{"priority", priority}};
        return info;
    }

    void TD3Agent::update_actor(const torch::Tensor &obs) {
        policy_optimizer->zero_grad();
        auto act = this->policy_net.forward(obs);
        auto q = this->q_network.forward(obs, act, true);
        auto policy_loss = -torch::mean(q);
        policy_loss.backward();
        policy_optimizer->step();
        m_logger->store("LossPi", policy_loss.item<float>());
    }

    torch::Tensor
    TD3Agent::compute_priority(const torch::Tensor &obs, const torch::Tensor &act, const torch::Tensor &next_obs,
                               const torch::Tensor &rew, const torch::Tensor &done) {
        torch::NoGradGuard no_grad;
        auto target_q_values = this->compute_target_q(next_obs, rew, done);
        auto q_values = this->q_network.forward(obs, act, true);
        return torch::abs(q_values - target_q_values);
    }

    void TD3Agent::set_grad(const str_to_tensor_list &grads) {
        auto q_grad = grads.at("q_grads");
        for (size_t i = 0; i < q_grad.size(); i++) {
            this->q_network.ptr()->parameters().at(i).mutable_grad() = q_grad.at(i);
        }
        if (grads.contains("p_grads")) {
            auto p_grad = grads.at("p_grads");
            for (size_t i = 0; i < p_grad.size(); i++) {
                this->policy_net.ptr()->parameters().at(i).mutable_grad() = p_grad.at(i);
            }
        }
    }

    std::pair<str_to_tensor_list, str_to_tensor>
    TD3Agent::compute_grad(const torch::Tensor &obs, const torch::Tensor &act, const torch::Tensor &next_obs,
                           const torch::Tensor &rew, const torch::Tensor &done,
                           const std::optional<torch::Tensor> &importance_weights, bool update_target) {
        // compute q loss
        auto target_q_values = this->compute_target_q(next_obs, rew, done);
        q_optimizer->zero_grad();
        auto q_values = this->q_network.forward(obs, act, false); // (num_ensemble, None)
        auto q_values_loss = 0.5 * torch::square(torch::unsqueeze(target_q_values, 0) - q_values);
        q_values_loss = torch::sum(q_values_loss, 0);  // (None,)
        // apply importance weights
        if (importance_weights != std::nullopt) {
            q_values_loss = q_values_loss * importance_weights.value();
        }
        q_values_loss = torch::mean(q_values_loss);
        auto grads = torch::autograd::grad({q_values_loss}, this->q_network.ptr()->parameters());

        str_to_tensor_list output{
                {"q_grads", grads}
        };
        auto priority = torch::abs(std::get<0>(torch::min(q_values, 0)) - target_q_values).detach().cpu();
        str_to_tensor info{{"priority", priority}};

        for (int i = 0; i < 2; i++) {
            m_logger->store(fmt::format("Q{}Vals", i + 1),
                            rlu::nn::convert_tensor_to_flat_vector<float>(q_values.index({0}).detach()));
        }
        m_logger->store("LossQ", q_values_loss.item<float>());

        // compute actor loss
        if (update_target) {
            auto current_act = this->policy_net.forward(obs);
            auto q = this->q_network.forward(obs, current_act, true);
            auto policy_loss = -torch::mean(q);
            auto policy_grads = torch::autograd::grad({policy_loss}, this->policy_net.ptr()->parameters());
            output["p_grads"] = policy_grads;
            m_logger->store("LossPi", policy_loss.item<float>());
        }

        return std::make_pair(output, info);
    }

    void TD3Agent::update_step(bool update_target) {
        q_optimizer->step();
        if (update_target) {
            policy_optimizer->step();
            this->update_target_q(true);
            this->update_target_policy(true);
        }
    }
}