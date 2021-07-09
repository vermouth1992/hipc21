//
// Created by chi on 7/2/21.
//

#include "include/gym/gym.h"
#include "dqn.h"
#include "cxxopts.hpp"
#include "functional.h"
#include <iostream>

cxxopts::ParseResult
parse(int argc, char *argv[]) {
    try {
        cxxopts::Options options(argv[0], " - Training DQN agents");
        options
                .positional_help("[optional args]")
                .show_positional_help();

        options
                .allow_unrecognised_options()
                .add_options()
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
                        ("alpha", "Prioritized replay buffer priority exponent",
                         cxxopts::value<float>()->default_value("0.6"))
                        ("initial_beta", "Prioritized replay buffer initial beta",
                         cxxopts::value<float>()->default_value("0.6"))
                        ("replay_size", "Size of the replay buffer",
                         cxxopts::value<int64_t>()->default_value("1000000"))
                        ("mlp_hidden", "Size of the MLP hidden layer", cxxopts::value<int64_t>()->default_value("128"))
                        ("double_q", "Double Q learning", cxxopts::value<bool>()->default_value("true"))
                        ("gamma", "discount factor", cxxopts::value<float>()->default_value("0.99"))
                        ("q_lr", "learning rate", cxxopts::value<float>()->default_value("0.001"))
                        ("tau", "polyak averaging of target network", cxxopts::value<float>()->default_value("0.005"))
                        ("epsilon_greedy", "exploration rate", cxxopts::value<float>()->default_value("0.1"))
                        ("device", "Pytorch device", cxxopts::value<std::string>()->default_value("cpu"))
                        ("num_actors", "Number of parallel actors", cxxopts::value<int64_t>()->default_value("1"));

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


int main(int argc, char **argv) {
    StopWatcher watcher;
    auto result = parse(argc, argv);
    try {
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

        int64_t num_actors = result["num_actors"].as<int64_t>();

        if (num_actors < 0) {
            boost::shared_ptr<Gym::Client> client = Gym::client_create("127.0.0.1", 5000);
            watcher.start();
            train_dqn(client,
                      result["env_id"].as<std::string>(),
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
                      result["alpha"].as<float>(),
                      result["initial_beta"].as<float>(),
                      result["replay_size"].as<int64_t>(),
                      result["mlp_hidden"].as<int64_t>(),
                      result["double_q"].as<bool>(),
                      result["gamma"].as<float>(),
                      result["q_lr"].as<float>(),
                      result["tau"].as<float>(),
                      result["epsilon_greedy"].as<float>(),
                      device);
        } else {
            std::vector<boost::shared_ptr<Gym::Client>> clients;
            clients.reserve(num_actors);
            for (int i = 0; i < num_actors; ++i) {
                clients.push_back(Gym::client_create("127.0.0.1", 5000 + i));
            }
            watcher.start();
            train_dqn_parallel(clients,
                               result["env_id"].as<std::string>(),
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
                               result["alpha"].as<float>(),
                               result["initial_beta"].as<float>(),
                               result["replay_size"].as<int64_t>(),
                               result["mlp_hidden"].as<int64_t>(),
                               result["double_q"].as<bool>(),
                               result["gamma"].as<float>(),
                               result["q_lr"].as<float>(),
                               result["tau"].as<float>(),
                               result["epsilon_greedy"].as<float>(),
                               device,
                               num_actors);
        }

        watcher.stop();
        std::cout << "Total execution time: " << watcher.seconds() << " s" << std::endl;

    } catch (const std::exception &e) {
        fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }

    return 0;
}