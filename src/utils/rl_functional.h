//
// Created by chi on 7/1/21.
//

#ifndef HIPC21_RL_FUNCTIONAL_H
#define HIPC21_RL_FUNCTIONAL_H

#include <chrono>
#include <utility>
#include <torch/torch.h>
#include <string>
#include <vector>


void hard_update(const torch::nn::Module &target, const torch::nn::Module &source);

void soft_update(const torch::nn::Module &target, const torch::nn::Module &source, float tau);


#endif //HIPC21_RL_FUNCTIONAL_H
