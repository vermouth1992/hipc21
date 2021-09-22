//
// Created by Chi Zhang on 8/5/21.
//

#include "agent/dqn.h"

namespace rlu::agent {

    DQN::DQN(const Gym::Space &obs_space,
             const Gym::Space &act_space,
             int64_t mlp_hidden, float q_lr, float gamma, float tau, bool double_q, float epsilon_greedy) :
            OffPolicyAgent(tau, q_lr, gamma),
            m_act_dim(act_space.discreet_n),
            m_double_q(double_q),
            m_epsilon_greedy(epsilon_greedy) {
        M_Assert(act_space.type == Gym::Space::DISCRETE, "Only support discrete action space");

        if (obs_space.box_shape.size() == 1) {
            int64_t obs_dim = obs_space.box_shape[0];
            int64_t act_dim = act_space.discreet_n;
            this->q_network = register_module("q_network", rlu::nn::build_mlp(obs_dim, act_dim, mlp_hidden));
            this->target_q_network = register_module("target_q_network",
                                                     rlu::nn::build_mlp(obs_dim, act_dim, mlp_hidden));
        } else {
            throw std::runtime_error("Unsupported observation space.");
        }

        this->q_optimizer = std::make_unique<torch::optim::Adam>(q_network.ptr()->parameters(),
                                                                 torch::optim::AdamOptions(this->q_lr));
        update_target_q(false);
    }

    str_to_tensor DQN::train_step(const torch::Tensor &obs, const torch::Tensor &act, const torch::Tensor &next_obs,
                                  const torch::Tensor &rew, const torch::Tensor &done,
                                  const std::optional<torch::Tensor> &importance_weights, bool update_target) {

        // compute target values
        torch::Tensor target_q_values = this->compute_next_obs_q(next_obs, rew, done);
        spdlog::debug("After computing target_q_values");
        q_optimizer->zero_grad();
        auto q_values = this->q_network.forward(obs);
        spdlog::debug("After forward");
        q_values = torch::gather(q_values, 1, act.unsqueeze(1)).squeeze(1); // (None,)
        auto loss = torch::square(q_values - target_q_values); // (None,)
        if (importance_weights != std::nullopt) {
            loss = loss * importance_weights.value();
        }
        loss = torch::mean(loss);
        spdlog::debug("After computing loss");
        loss.backward();
        spdlog::debug("After loss backward");
        q_optimizer->step();
        spdlog::debug("After optimizer step");

        if (update_target) {
            this->update_target_q(true);
        }

        spdlog::debug("After target update");

        str_to_tensor log_data{
                {"priority", torch::abs(q_values - target_q_values).detach()}
        };

        // logging
        m_logger->store("QVals", rlu::nn::convert_tensor_to_flat_vector<float>(q_values));
        m_logger->store("LossQ", loss.item<float>());

        spdlog::debug("After logging");

        return log_data;
    }

    torch::Tensor DQN::act_single(const torch::Tensor &obs, bool exploration) {
        if (exploration) {
            float rand_num = torch::rand({}).item().toFloat();
            if (rand_num > m_epsilon_greedy) {
                // execute inference
                return act_test_single(obs);
            } else {
                // random sample
                return torch::randint(m_act_dim, {}, torch::TensorOptions().dtype(torch::kInt64));
            }
        } else {
            // execute inference
            return act_test_single(obs);
        }
    }

    torch::Tensor DQN::act_test_single(const torch::Tensor &obs) {
        torch::NoGradGuard no_grad;
        auto obs_batch = obs.unsqueeze(0);
        auto q_values = this->q_network.forward(obs_batch); // shape (None, act_dim)
        auto act_batch = std::get<1>(torch::max(q_values, -1));
        return act_batch.index({0});
    }

    void DQN::log_tabular() {
        m_logger->log_tabular("QVals", std::nullopt, true);
        m_logger->log_tabular("LossQ", std::nullopt, false, true);
    }

    str_to_tensor_list
    DQN::compute_grad(const torch::Tensor &obs, const torch::Tensor &act, const torch::Tensor &next_obs,
                      const torch::Tensor &rew, const torch::Tensor &done,
                      const std::optional<torch::Tensor> &importance_weights,
                      __attribute__((unused)) bool update_target) {
        torch::Tensor target_q_values = this->compute_next_obs_q(next_obs, rew, done);
        auto q_values = this->q_network.forward(obs);
        q_values = torch::gather(q_values, 1, act.unsqueeze(1)).squeeze(1); // (None,)
        auto loss = torch::square(q_values - target_q_values); // (None,)
        if (importance_weights != std::nullopt) {
            loss = loss * importance_weights.value();
        }
        loss = torch::mean(loss);
        auto grads = torch::autograd::grad({loss}, this->q_network.ptr()->parameters());
        str_to_tensor_list output{
                {"q_grads", grads}
        };

        // logging. Need synchronization
        m_logger->store("QVals", rlu::nn::convert_tensor_to_flat_vector<float>(q_values));
        m_logger->store("LossQ", loss.item<float>());

        return output;
    }

    void DQN::set_grad(const str_to_tensor_list &grads) {
        auto q_grad = grads.at("q_grads");
        for (size_t i = 0; i < q_grad.size(); i++) {
            this->q_network.ptr()->parameters().at(i).mutable_grad() = q_grad.at(i);
        }
    }

    void DQN::update_step(bool update_target) {
        q_optimizer->step();
        if (update_target) {
            this->update_target_q(true);
        }
    }

    torch::Tensor
    DQN::compute_next_obs_q(const torch::Tensor &next_obs, const torch::Tensor &rew, const torch::Tensor &done) {
        // compute target values
        torch::NoGradGuard no_grad;
        torch::Tensor target_q_values = this->target_q_network.forward(next_obs); // shape (None, act_dim)
        if (m_double_q) {
            auto target_actions = std::get<1>(
                    torch::max(this->q_network.forward(next_obs), -1)); // shape (None,)
            target_q_values = torch::gather(target_q_values, 1, target_actions.unsqueeze(1)).squeeze(1);
        } else {
            target_q_values = std::get<0>(torch::max(target_q_values, -1));
        }
        target_q_values = rew + gamma * (1. - done) * target_q_values;
        return target_q_values;
    }

    torch::Tensor
    DQN::compute_priority(const torch::Tensor &obs, const torch::Tensor &act, const torch::Tensor &next_obs,
                          const torch::Tensor &rew, const torch::Tensor &done) {
        torch::NoGradGuard no_grad;
        torch::Tensor target_q_values = this->compute_next_obs_q(next_obs, rew, done);
        auto q_values = this->q_network.forward(obs);
        q_values = torch::gather(q_values, 1, act.unsqueeze(1)).squeeze(1); // (None,)
        return torch::abs(q_values - target_q_values);
    }

}