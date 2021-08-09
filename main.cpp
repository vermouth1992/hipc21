//
// Created by chi on 7/2/21.
//


#include "agent/dqn.h"
#include "agent/td3.h"
#include "trainer/off_policy_trainer.h"
#include "fmt/core.h"
#include "cxxopts.hpp"

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

std::shared_ptr<OffPolicyAgent> create_agent(const std::function<std::shared_ptr<Gym::Environment>()> &env_fn,
                                             const std::string &algorithm) {
    auto env = env_fn();
    std::shared_ptr<OffPolicyAgent> agent;
    if (algorithm == "dqn") {
        agent = std::make_shared<DQN>(*env->observation_space(),
                                      *env->action_space());
    } else if (algorithm == "td3") {
        agent = std::make_shared<TD3Agent>(*env->observation_space(),
                                           *env->action_space());
    } else if (algorithm == "sac") {

    } else {
        throw std::runtime_error(fmt::format("Unknown algorithm {}", algorithm));
    }
    env->close();
    return agent;
}

int main(int argc, char **argv) {
    // remove the second argv
    if (argc < 2) throw std::runtime_error("Must specify the algorithm");
    std::string algorithm(argv[1]);
    char **real_argv = new char *[argc - 1];
    real_argv[0] = argv[0];
    for (int i = 2; i < argc; i++) {
        real_argv[i - 1] = argv[i];
    }

    try {
        auto result = parse(argc - 1, real_argv);
        // device name
        std::string device_name = result["device"].as<std::string>();
        auto device = get_torch_device(device_name);
        std::shared_ptr<Gym::Client> client = Gym::client_create("127.0.0.1", 5000);
        std::string env_id = result["env_id"].as<std::string>();
        std::shared_ptr<Gym::Environment> env = client->make(env_id);

        std::function<std::shared_ptr<Gym::Environment>()> env_fn = [&client, &env_id]() {
            return client->make(env_id);
        };

        std::function<std::shared_ptr<OffPolicyAgent>()> agent_fn = [&env_fn, &algorithm]() {
            return create_agent(env_fn, algorithm);
        };

        OffPolicyTrainer trainer(env_fn,
                                 agent_fn,
                                 result["epochs"].as<int64_t>(),
                                 result["steps_per_epoch"].as<int64_t>(),
                                 result["start_steps"].as<int64_t>(),
                                 result["update_after"].as<int64_t>(),
                                 result["update_every"].as<int64_t>(),
                                 result["update_per_step"].as<int64_t>(),
                                 result["policy_delay"].as<int64_t>(),
                                 result["num_test_episodes"].as<int64_t>(),
                                 device,
                                 result["seed"].as<int64_t>()
        );

        trainer.setup_logger(std::nullopt, "data");
        trainer.setup_replay_buffer(result["replay_size"].as<int64_t>(), result["batch_size"].as<int64_t>());
        trainer.train();

    } catch (const std::exception &e) {
        fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }

    delete[]real_argv;

    return 0;
}