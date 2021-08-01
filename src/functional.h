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

struct Mlp : torch::nn::Module {
    torch::nn::Linear linear1;
    torch::nn::Linear linear2;
    torch::nn::Linear linear3;

    Mlp(int input_size, int output_size, int mlp_hidden) :
            linear1(register_module("linear1", torch::nn::Linear(input_size, mlp_hidden))),
            linear2(register_module("linear2", torch::nn::Linear(mlp_hidden, mlp_hidden))),
            linear3(register_module("linear3", torch::nn::Linear(mlp_hidden, output_size))) {
    }

    torch::Tensor forward(torch::Tensor x) {
        x = linear1->forward(x);
        x = torch::nn::functional::relu(x);
        x = linear2->forward(x);
        x = torch::nn::functional::relu(x);
        x = linear3->forward(x);
        return x;
    }
};

class Scheduler {
    virtual double value(int64_t t) = 0;
};

class ConstantScheduler : public Scheduler {
public:
    explicit ConstantScheduler(double value) : m_value(value) {

    }

    double value(__attribute__((unused)) int64_t t) override {
        return m_value;
    }

private:
    double m_value;
};


class LinearSchedule : public Scheduler {
public:
    LinearSchedule(int64_t scheduleTimesteps, double finalP, double initialP)
            : schedule_timesteps(scheduleTimesteps),
              final_p(finalP),
              initial_p(initialP) {

    }

    double value(int64_t t) override {
        double fraction = std::min((double) t / (double) schedule_timesteps, 1.0);
        return initial_p + fraction * (final_p - initial_p);
    }

private:
    int64_t schedule_timesteps;
    double final_p;
    double initial_p;
};


class PiecewiseSchedule : public Scheduler {
    // TODO
};

class StopWatcher {
public:
    explicit StopWatcher(std::string name) : m_elapsed(0), m_name(std::move(name)) {

    }

    explicit StopWatcher() : m_elapsed(0), m_name("default") {

    }

    std::string name() const {
        return m_name;
    }

    void reset() {
        m_elapsed = 0;
    }

    void start() {
        m_start_time = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        m_elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - m_start_time).count();
    }

    int64_t nanoseconds() const {
        return m_elapsed;
    }

    double microseconds() const {
        return (double) nanoseconds() / 1000.;
    }

    double milliseconds() const {
        return (double) nanoseconds() / 1000000.;
    }

    double seconds() const {
        return (double) nanoseconds() / 1000000000.;
    }


private:
    int64_t m_elapsed;
    std::string m_name;
    std::chrono::high_resolution_clock::time_point m_start_time;
};

#endif //HIPC21_FUNCTIONAL_H
