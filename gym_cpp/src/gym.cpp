//
// Created by chi on 10/20/21.
//

#include "gym_cpp/gym.h"

namespace gym {
    std::shared_ptr<env::Env>
    make(const std::string &env_name, const std::optional<std::unordered_map<std::string, std::any>> &kwargs,
         const std::string &protocol) {
        if (protocol == "ZMQ") {
            auto output = std::make_shared<env::ZMQEnv>(env_name);
            return output;
        } else {
            throw std::runtime_error("Unimplemented protocol");
        }
    }
}