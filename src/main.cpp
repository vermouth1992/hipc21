//
// Created by chi on 7/2/21.
//


#include "agent/agent.h"
#include "trainer/trainer.h"
#include "trainer/off_policy_trainer_fpga.h"
#include "cxxopts.hpp"
#include "spdlog/spdlog.h"

static cxxopts::ParseResult parse(int argc, char *argv[]) {
    try {
        cxxopts::Options options(argv[0], " - Training Off-policy agents");
        options.positional_help("[optional args]").show_positional_help();
        options.allow_unrecognised_options().add_options()
                ("help", "Print help")
                ("env_id", "Environment id", cxxopts::value<std::string>())
                ("algorithm", "Algorithm", cxxopts::value<std::string>())
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
                ("device", "Pytorch device", cxxopts::value<std::string>()->default_value("cpu"))
                ("num_actors", "Number of actors", cxxopts::value<int64_t>()->default_value("0"))
                ("num_learners", "Number of learners", cxxopts::value<int64_t>()->default_value("0"))
                ("bitstream", "Path to the bitstream", cxxopts::value<std::string>()->default_value(" "))
                ("logging_level", "Logging level", cxxopts::value<int>()->default_value("2"));

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

std::shared_ptr<rlu::agent::OffPolicyAgent>
create_agent(const std::function<std::shared_ptr<Gym::Environment>()> &env_fn,
             const std::string &algorithm) {
    auto env = env_fn();
    std::shared_ptr<rlu::agent::OffPolicyAgent> agent;
    if (algorithm == "dqn") {
        agent = std::make_shared<rlu::agent::DQN>(*env->observation_space(),
                                                  *env->action_space());
    } else if (algorithm == "td3") {
        agent = std::make_shared<rlu::agent::TD3Agent>(*env->observation_space(),
                                                       *env->action_space());
    } else if (algorithm == "sac") {

    } else {
        throw std::runtime_error(fmt::format("Unknown algorithm {}", algorithm));
    }
    env->close();
    return agent;
}

int main(int argc, char **argv) {
    try {
        auto result = parse(argc, argv);
        int logging_level = result["logging_level"].as<int>();
        spdlog::set_level(static_cast<spdlog::level::level_enum>(logging_level));
        torch::manual_seed(result["seed"].as<int64_t>());
        // device name
        std::string algorithm(result["algorithm"].as<std::string>());
        std::string device_name(result["device"].as<std::string>());
        std::string env_id = result["env_id"].as<std::string>();

        int64_t num_actors = result["num_actors"].as<int64_t>();
        int64_t num_learners = result["num_learners"].as<int64_t>();

        int port = 5000;

        // get environment function
        std::function<std::shared_ptr<Gym::Environment>()> env_fn = [&port, &env_id]() {
            spdlog::info("Creating environment with port {}", port);
            std::shared_ptr<Gym::Client> client = Gym::client_create("127.0.0.1", port);
            port += 1;
            return client->make(env_id);
        };

        // get agent function
        std::function<std::shared_ptr<rlu::agent::OffPolicyAgent>()> agent_fn = [&env_fn, &algorithm]() {
            spdlog::info("Creating agent");
            return create_agent(env_fn, algorithm);
        };

        std::shared_ptr<rlu::trainer::OffPolicyTrainer> trainer;

        if (num_actors <= 0 || num_learners <= 0) {
            spdlog::info("Running sequential trainer");
            auto device = rlu::ptu::get_torch_device(device_name);
            trainer = std::make_shared<rlu::trainer::OffPolicyTrainer>(env_fn,
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
                                                                       result["seed"].as<int64_t>());
        } else if (device_name == "cpu" || device_name == "gpu") {
            spdlog::info("Running parallel trainer");
            auto device = rlu::ptu::get_torch_device(device_name);
            trainer = std::make_shared<rlu::trainer::OffPolicyTrainerParallel>(
                    env_fn,
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
                    result["seed"].as<int64_t>(),
                    num_actors,
                    num_learners);
        } else if (device_name == "fpga") {
            spdlog::info("Running fpga trainer");
            // fpga trainer
            trainer = std::make_shared<rlu::trainer::OffPolicyTrainerFPGA>(
                    env_fn,
                    agent_fn,
                    result["epochs"].as<int64_t>(),
                    result["steps_per_epoch"].as<int64_t>(),
                    result["start_steps"].as<int64_t>(),
                    result["update_after"].as<int64_t>(),
                    result["update_every"].as<int64_t>(),
                    result["update_per_step"].as<int64_t>(),
                    result["policy_delay"].as<int64_t>(),
                    result["num_test_episodes"].as<int64_t>(),
                    result["seed"].as<int64_t>(),
                    num_actors,
                    result["bitstream"].as<std::string>()
            );
        } else {
            throw std::runtime_error(fmt::format("Unknown device name {}", device_name));
        }

        trainer->setup_environment();
        trainer->setup_logger(std::nullopt, "data");
        trainer->setup_replay_buffer(result["replay_size"].as<int64_t>(), result["batch_size"].as<int64_t>());
        trainer->train();

    } catch (const std::exception &e) {
        fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }

    return 0;
}