//
// Created by chi on 10/20/21.
//

#ifndef GYM_CPP_GYM_H
#define GYM_CPP_GYM_H

#include "gym_cpp/envs/env.h"
#include <any>
#include <optional>

namespace gym {
    std::shared_ptr<env::Env> make(const std::string &env_name,
                                   const std::optional<std::unordered_map<std::string, std::any>> &kwargs = std::nullopt,
                                   const std::string &protocol = "ZMQ");
}


#endif //GYM_CPP_GYM_H
