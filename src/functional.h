//
// Created by chi on 7/1/21.
//

#ifndef HIPC21_FUNCTIONAL_H
#define HIPC21_FUNCTIONAL_H

#include <chrono>
#include <utility>
#include <torch/torch.h>
#include <string>
#include <vector>

template<class T>
static std::pair<float, float> compute_mean_std(const std::vector<T> &v) {
    float sum = std::accumulate(v.begin(), v.end(), 0.0);
    float mean = sum / v.size();

    float sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    float stddev = std::sqrt(sq_sum / v.size() - mean * mean);
    return std::make_pair(mean, stddev);
}

static void hard_update(const torch::nn::Module &target, const torch::nn::Module &source) {
    {
        torch::NoGradGuard no_grad;
        for (uint i = 0; i < target.parameters().size(); i++) {
            auto target_param = target.parameters()[i];
            auto param = source.parameters()[i];
            target_param.data().copy_(param.data());
        }
    }
}

static void soft_update(const torch::nn::Module &target, const torch::nn::Module &source, float tau) {
    {
        torch::NoGradGuard no_grad;
        for (uint i = 0; i < target.parameters().size(); i++) {
            auto target_param = target.parameters()[i];
            auto param = source.parameters()[i];
            target_param.data().copy_(target_param.data() * (1.0 - tau) + param.data() * tau);
        }
    }
}




#endif //HIPC21_FUNCTIONAL_H
