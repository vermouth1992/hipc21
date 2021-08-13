#ifndef __GYM_H__
#define __GYM_H__

#include <vector>
#include <string>
#include <random>
#include <torch/torch.h>
#include "nlohmann/json.hpp"
#include "fmt/format.h"

using nlohmann::json;

namespace Gym {

    struct Space {
        enum SpaceType {
            DISCRETE,
            BOX,
        } type;

        // Random vector that belong to this space
        [[nodiscard]] torch::Tensor sample() const {
            if (type == DISCRETE) {
                return torch::randint(discreet_n, {}, torch::TensorOptions().dtype(torch::kInt64));
            }
            assert(type == BOX);
            auto rand_num = torch::rand(box_shape, torch::TensorOptions().dtype(torch::kFloat32));
            return (box_high - box_low) * rand_num + box_low;
        }

        std::vector<int64_t> box_shape; // Similar to Caffe blob shape, for example { 64, 96, 3 } for 96x64 rgb image.
        torch::Tensor box_high;
        torch::Tensor box_low;

        int discreet_n;
    };

    struct State {
        torch::Tensor observation; // get observation_space() to make sense of this data
        float reward;
        bool done;
        bool timeout;
        std::string info;
    };

    class Environment {
    public:
        std::string env_id;

        virtual std::shared_ptr<Space> action_space() = 0;

        virtual std::shared_ptr<Space> observation_space() = 0;

        virtual void reset(State *save_initial_state_here) = 0;

        virtual void step(const torch::Tensor &action, bool render, State *save_state_here) = 0;

        virtual void close() = 0;
    };

    class Client {
    public:
        virtual std::shared_ptr<Environment> make(const std::string &name) = 0;
    };

    extern std::shared_ptr<Client> client_create(const std::string &addr, int port);

} // namespace

#endif // __GYM_H__
