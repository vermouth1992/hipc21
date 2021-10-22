//
// Created by chi on 10/20/21.
//

#include "gym_cpp/envs/env.h"

#include <utility>

namespace gym::env {
    torch::Tensor decode_base64_to_tensor(const std::string &str) {
        std::vector<char> f;
        // b64decode
        macaron::Base64::Decode(str, f);
        torch::Tensor x = torch::pickle_load(f).toTensor();
        return x;
    }

    std::string encode_tensor(const torch::Tensor &tensor) {
        std::vector<char> f = torch::pickle_save(tensor);
        // convert to base64
        std::string result = macaron::Base64::Encode(f);
        return result;
    }

    std::shared_ptr<space::Space> decode_space(const std::string &space) {
        auto j = json::parse(space);
        std::string dtype = j["type"].get<std::string>();
        if (dtype == "Box") {
            auto low = decode_base64_to_tensor(j["low"].get<std::string>());
            auto high = decode_base64_to_tensor(j["high"].get<std::string>());
            return std::make_shared<space::Box>(low, high);
        } else if (dtype == "Discrete") {
            auto n = j["n"].get<int64_t>();
            return std::make_shared<space::Discrete>(n);
        } else {
            throw std::runtime_error("Unknown type");
        }
    }

    Env::Env(std::string env_name) : env_name(std::move(env_name)) {

    }

    std::string Env::get_env_name() const {
        return env_name;
    }

    ZMQEnv::ZMQEnv(const std::string &env_name) :
            Env(env_name),
            temp_dir(fs::temp_directory_path() / "gym") {
        // create server
        create_server_and_client();
        create_env(env_name);
        // setup observation space and action space
        set_observation_space();
        set_action_space();
    }

    ZMQEnv::~ZMQEnv() {
        sock.close();
        server_proc->kill();
    }

    void ZMQEnv::reset(State &state) {
        json j;
        j["command"] = "reset";
        auto obs = this->query(j);
        state.observation = decode_base64_to_tensor(obs);
    }

    void ZMQEnv::step(const torch::Tensor &action, State &state) {
        auto j = encode_action(action);
        j["command"] = "step";
        auto output = json::parse(this->query(j));
        auto next_obs = decode_base64_to_tensor(output["next_obs"].get<std::string>());
        auto reward = output["reward"].get<float>();
        auto done = output["done"].get<bool>();
        auto info = output["info"];
        state.observation = next_obs;
        state.reward = reward;
        state.done = done;
        state.info = info;
        if (info.contains("timeout")) {
            state.timeout = info["timeout"].get<bool>();
        } else {
            state.timeout = false;
        }
    }

    void ZMQEnv::close() {
        // it is crucial to close the socket. Otherwise, the destructor of the context will hang.
        sock.close();
        server_proc->kill();
    }

    void ZMQEnv::seed(int64_t seed) {
        json j;
        j["command"] = "seed";
        j["seed"] = seed;
        auto message = this->query(j);
        assert(message == "done");
    }

    int64_t ZMQEnv::find_available_port() {
        int max_trial = 100;
        for (int i = 0; i < max_trial; ++i) {
            int64_t port = Random::get(0, 9999);
            auto path = fs::path(temp_dir) / std::to_string(port);
            if (!fs::exists(path)) {
                return port;
            }
        }
        throw std::runtime_error("Can't find available port in " + std::to_string(max_trial) + " trials");
        return 0;
    }

    std::string ZMQEnv::query(const json &j) {
        auto str_j = j.dump();
        sock.send(zmq::buffer(str_j));
        zmq::message_t reply;
        auto result = sock.recv(reply);
        (void) result;
        auto reply_str = reply.to_string();
        return reply_str;
    }

    void ZMQEnv::create_server_and_client() {
        int64_t port = find_available_port();
        auto python_path_env = std::getenv("PYTHON_EXECUTABLE");
        std::string python_path;
        if (python_path_env == nullptr) {
            python_path = "python";
        } else {
            python_path = std::string(python_path_env);
        }
        server_proc = std::make_shared<subprocess::Popen>(
                python_path + " -m gym_zmq.zeromq_gym_server -p " + std::to_string(port));
        // create client socket
        sock = zmq::socket_t(ctx, zmq::socket_type::req);
        address = "ipc://" + temp_dir + "/" + std::to_string(port);
        sock.connect(address);
    }

    void ZMQEnv::create_env(const std::string &env_name) {
        json j;
        j["command"] = "create";
        j["env_name"] = env_name;
        auto output = this->query(j);
        assert(output == "done");
    }

    void ZMQEnv::set_observation_space() {
        json j;
        j["command"] = "observation_space";
        auto output = this->query(j);
        observation_space = decode_space(output);
    }

    void ZMQEnv::set_action_space() {
        json j;
        j["command"] = "action_space";
        auto output = this->query(j);
        action_space = decode_space(output);
    }

    json ZMQEnv::encode_action(const torch::Tensor &action) {
        json j;
        if (action_space->get_type() == space::Type::Discrete_t) {
            j["action"] = action.item<int64_t>();
        } else {
            j["action"] = encode_tensor(action);
        }
        return j;
    }


}