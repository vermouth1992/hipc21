//
// Created by chi on 10/20/21.
//

#include "gym_cpp/spaces/space.h"

namespace gym::space {

    Box::Box(torch::Tensor low, torch::Tensor high) : low(std::move(low)), high(std::move(high)) {

    }

    torch::Tensor Box::sample() const {
        auto result = torch::rand(low.sizes(), this->gen);
        return result * (high - low) + low;
    }

    bool Box::contains(const torch::Tensor &x) const {
        auto result = torch::logical_and(x.greater_equal(low), x.less_equal(high));
        return result.all().item<bool>();
    }

    torch::IntArrayRef Box::get_shape() const {
        return low.sizes();
    }

    const torch::Tensor &Box::get_low() const {
        return low;
    }

    const torch::Tensor &Box::get_high() const {
        return high;
    }

    Discrete::Discrete(int64_t n) : n(n) {

    }

    torch::Tensor Discrete::sample() const {
        auto result = torch::randint(n, {}, this->gen);
        return result;
    }

    bool Discrete::contains(const torch::Tensor &x) const {
        auto result = torch::logical_and(x.greater_equal(0), x.less(n));
        return result.all().item<bool>();
    }

    int64_t Discrete::get_n() const {
        return n;
    }
}