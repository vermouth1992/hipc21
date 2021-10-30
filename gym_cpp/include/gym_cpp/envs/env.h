//
// Created by chi on 10/19/21.
//

#ifndef GYM_CPP_ENV_H
#define GYM_CPP_ENV_H

#include <torch/torch.h>
#include <unordered_map>
#include <filesystem>
#include <string>
#include <random>
#include <cstdlib>
#include <zmq.hpp>

#include "gym_cpp/spaces/space.h"

#include "nlohmann/json.hpp"
#include "third_party/effolkronium/random.hpp"
#include "third_party/subprocess.hpp"
#include "third_party/base64.h"

using json = nlohmann::json;
namespace fs = std::filesystem;
using Random = effolkronium::random_static;

namespace gym::env {
    struct State {
        torch::Tensor observation;
        float reward;
        bool done;
        bool timeout;
        json info;
    };

    class Env {
    public:
        explicit Env(std::string env_name);

        virtual ~Env();

        virtual void reset(State &state) = 0;

        virtual void step(const torch::Tensor &action, State &state) = 0;

        virtual void close() = 0;

        virtual void seed(int64_t seed) = 0;

        [[nodiscard]] std::string get_env_name() const;

        std::shared_ptr<space::Space> observation_space;
        std::shared_ptr<space::Space> action_space;

    protected:
        const std::string env_name;

    };


    class ZMQEnv : public Env {
    public:
        explicit ZMQEnv(const std::string &env_name);

        ~ZMQEnv();

        void reset(State &state) override;

        void step(const torch::Tensor &action, State &state) override;

        void close() override;

        void seed(int64_t seed) override;

    private:
        int64_t find_available_port();

        std::string query(const json &j);

        void create_server_and_client();

        void create_env(const std::string &env_name);

        void set_observation_space();

        void set_action_space();

        json encode_action(const torch::Tensor &action);

        const std::string temp_dir;
        std::shared_ptr<subprocess::Popen> server_proc;
        zmq::socket_t sock;
        zmq::context_t ctx;
        std::string address;
    };
}

#endif //GYM_CPP_ENV_H
