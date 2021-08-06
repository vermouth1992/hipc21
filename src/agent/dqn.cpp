//
// Created by Chi Zhang on 8/5/21.
//

#include "nn/functional.h"
#include "dqn.h"
#include "cxxopts.hpp"
#include "fmt/ranges.h"

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
        this->q_network = register_module("q_network", build_mlp(obs_dim, act_dim, mlp_hidden));
        this->target_q_network = register_module("target_q_network", build_mlp(obs_dim, act_dim, mlp_hidden));
    } else {
        throw std::runtime_error("Unsupported observation space.");
    }

    this->q_optimizer = std::make_unique<torch::optim::Adam>(q_network.ptr()->parameters(),
                                                             torch::optim::AdamOptions(this->q_lr));
    update_target(false);
}

OffPolicyAgent::str_to_tensor
DQN::train_step(const torch::Tensor &obs, const torch::Tensor &act, const torch::Tensor &next_obs,
                const torch::Tensor &rew, const torch::Tensor &done,
                const std::optional<torch::Tensor> &importance_weights) {

    // compute target values
    torch::Tensor target_q_values;
    {
        torch::NoGradGuard no_grad;
        target_q_values = this->target_q_network.forward(next_obs); // shape (None, act_dim)

        if (m_double_q) {
            auto target_actions = std::get<1>(torch::max(this->q_network.forward(next_obs), -1)); // shape (None,)
            target_q_values = torch::gather(target_q_values, 1, target_actions.unsqueeze(1)).squeeze(1);
        } else {
            target_q_values = std::get<0>(torch::max(target_q_values, -1));
        }
        target_q_values = rew + gamma * (1. - done) * target_q_values;
    }
    q_optimizer->zero_grad();
    auto q_values = this->q_network.forward(obs);
    q_values = torch::gather(q_values, 1, act.unsqueeze(1)).squeeze(1); // (None,)
    auto loss = torch::square(q_values - target_q_values); // (None,)
    if (importance_weights != std::nullopt) {
        loss = loss * importance_weights.value();
    }
    loss = torch::mean(loss);
    loss.backward();
    q_optimizer->step();

    str_to_tensor log_data{
            {"abs_delta_q", torch::abs(q_values - target_q_values).detach()}
    };

    // logging
    m_logger->store({
                            {"QVals", tensor_to_vector(q_values)},
                            {"LossQ", std::vector<float>{loss.item<float>()}}
                    });

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
    {
        torch::NoGradGuard no_grad;
        auto obs_batch = obs.unsqueeze(0);
        auto q_values = this->q_network.forward(obs_batch); // shape (None, act_dim)
        auto act_batch = std::get<1>(torch::max(q_values, -1));
        return act_batch.index({0});
    }
}

void DQN::log_tabular() {
    m_logger->log_tabular("QVals", std::nullopt, true);
    m_logger->log_tabular("LossQ", std::nullopt, false, true);
}

torch::Tensor DQN::act_batch(const torch::Tensor &obs, bool exploration) {
    // not implemented for now.
    return {};
}


static cxxopts::ParseResult parse(int argc, char *argv[]) {
    try {
        cxxopts::Options options(argv[0], " - Training DQN agents");
        options.positional_help("[optional args]").show_positional_help();
        options.allow_unrecognised_options().add_options()
                ("help", "Print help")
                ("env_id", "Environment id", cxxopts::value<std::string>())
                ("epochs", "Number of epochs", cxxopts::value<int64_t>()->default_value("100"))
                ("steps_per_epoch", "Number steps/epoch", cxxopts::value<int64_t>()->default_value("1000"))
                ("start_steps", "Number of steps that take random actions",
                 cxxopts::value<int64_t>()->default_value("1000"))
                ("update_after", "Number of steps before update",
                 cxxopts::value<int64_t>()->default_value("500"))
                ("update_every", "Number of steps between updates",
                 cxxopts::value<int64_t>()->default_value("1"))
                ("update_per_step", "Number of updates per step", cxxopts::value<int64_t>()->default_value("1"))
                ("policy_delay", "Number of steps for target update",
                 cxxopts::value<int64_t>()->default_value("1"))
                ("batch_size", "Size of one batch to perform update",
                 cxxopts::value<int64_t>()->default_value("100"))
                ("num_test_episodes", "Number of test episodes", cxxopts::value<int64_t>()->default_value("10"))
                ("seed", "Random seed", cxxopts::value<int64_t>()->default_value("1"))
                ("replay_size", "Size of the replay buffer",
                 cxxopts::value<int64_t>()->default_value("1000000"))
                ("mlp_hidden", "Size of the MLP hidden layer", cxxopts::value<int64_t>()->default_value("128"))
                ("double_q", "Double Q learning", cxxopts::value<bool>()->default_value("true"))
                ("gamma", "discount factor", cxxopts::value<float>()->default_value("0.99"))
                ("q_lr", "learning rate", cxxopts::value<float>()->default_value("0.001"))
                ("tau", "polyak averaging of target network", cxxopts::value<float>()->default_value("0.005"))
                ("epsilon_greedy", "exploration rate", cxxopts::value<float>()->default_value("0.1"))
                ("device", "Pytorch device", cxxopts::value<std::string>()->default_value("cpu"));

        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            exit(0);
        }

        return result;

    } catch (const cxxopts::OptionException &e) {
        std::cout << "error parsing options: " << e.what() << std::endl;
        exit(1);
    }
}


int dqn_main(int argc, char **argv) {
    auto result = parse(argc, argv);
    // device name
    std::string device_name = result["device"].as<std::string>();
    torch::DeviceType device_type;
    if (device_name == "gpu" && torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);
    std::shared_ptr<Gym::Client> client = Gym::client_create("127.0.0.1", 5000);
    const std::string env_id = result["env_id"].as<std::string>();
    std::shared_ptr<Gym::Environment> env = client->make(env_id);
    std::shared_ptr<Gym::Environment> test_env = client->make(env_id);
    // construct agent
    std::shared_ptr<OffPolicyAgent> agent;
    auto obs_shape = env->observation_space()->box_shape;
    if (obs_shape.size() == 1) {
        // low dimensional env
        agent = std::make_shared<MlpDQN>(obs_shape.at(0),
                                         env->action_space()->discreet_n,
                                         result["mlp_hidden"].as<int64_t>(),
                                         result["double_q"].as<bool>(),
                                         result["q_lr"].as<float>(),
                                         result["gamma"].as<float>(),
                                         result["tau"].as<float>(),
                                         result["epsilon_greedy"].as<float>());
    } else if (obs_shape.size() == 3) {
        // image env
        agent = std::make_shared<AtariDQN>(env->action_space()->discreet_n,
                                           result["double_q"].as<bool>(),
                                           result["epsilon_greedy"].as<float>());
    } else {
        throw std::runtime_error(fmt::format("Unsupported observation shape {}", obs_shape.size()));
    }

    agent->to(device);

    off_policy_trainer(env,
                       test_env,
                       std::nullopt,
                       "data",
                       result["epochs"].as<int64_t>(),
                       result["steps_per_epoch"].as<int64_t>(),
                       result["start_steps"].as<int64_t>(),
                       result["update_after"].as<int64_t>(),
                       result["update_every"].as<int64_t>(),
                       result["update_per_step"].as<int64_t>(),
                       result["policy_delay"].as<int64_t>(),
                       result["batch_size"].as<int64_t>(),
                       result["num_test_episodes"].as<int64_t>(),
                       result["seed"].as<int64_t>(),
                       result["replay_size"].as<int64_t>(),
                       agent,
                       device
    );
    return 0;
}
