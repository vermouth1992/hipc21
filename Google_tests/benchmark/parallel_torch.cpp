//
// Created by Chi Zhang on 9/28/21.
//

#include "gtest/gtest.h"
#include "nn/functional.h"
#include "utils/stop_watcher.h"


class BenchmarkParallelInference {
public:
    BenchmarkParallelInference() {
        for (int i = 0; i < num_threads; i++) {
            auto net = rlu::nn::build_mlp(input_dim, output_dim, mlp_hidden);
            net->to(device);
            this->networks.push_back(net);
        }
    }

    void run_serial() {
        // serial version
        watcher.reset();
        watcher.start();
        torch::Tensor result = torch::zeros({batch_size, output_dim}).to(device);
        torch::Tensor input = torch::ones({batch_size, input_dim}).to(device);
        for (int64_t i = 0; i < num_iterations; i++) {
            result += networks.at(0)->forward(input);
        }
        watcher.lap();
        std::cout << torch::sum(result) << std::endl;
        std::cout << "Serial version elapse " << watcher.seconds() << std::endl;
    }

    void run_parallel() {
        // parallel version
        std::vector<pthread_t> threads(num_threads);
        watcher.reset();
        watcher.start();
        for (int i = 0; i < num_threads; i++) {
            pthread_create(&threads.at(i), nullptr, &parallel_inference, this);
        }
        for (int i = 0; i < num_threads; i++) {
            pthread_join(threads.at(i), nullptr);
        }
        watcher.lap();
        std::cout << "Parallel version elapse " << watcher.seconds() << std::endl;
    }

    void run_parallel_network() {
        std::vector<pthread_t> threads(num_threads);
        watcher.reset();
        watcher.start();
        for (int i = 0; i < num_threads; i++) {
            auto data = std::make_pair(this, i);
            pthread_create(&threads.at(i), nullptr, &parallel_inference_v2, &data);
        }
        for (int i = 0; i < num_threads; i++) {
            pthread_join(threads.at(i), nullptr);
        }
        watcher.lap();
        std::cout << "Parallel version with different networks elapse " << watcher.seconds() << std::endl;
    }

private:
    static void *parallel_inference(void *ptr) {
        ((BenchmarkParallelInference *) ptr)->parallel_inference_thread();
        return nullptr;
    }

    static void *parallel_inference_v2(void *ptr) {
        auto *data = (std::pair<BenchmarkParallelInference *, int> *) ptr;
        data->first->parallel_inference_thread_v2(data->second);
        return nullptr;
    }

    void parallel_inference_thread() {
        torch::Tensor result = torch::zeros({batch_size, output_dim}).to(device);
        torch::Tensor input = torch::ones({batch_size, input_dim}).to(device);
        rlu::watcher::StopWatcher local_watcher;
        local_watcher.reset();
        local_watcher.start();
        for (int64_t i = 0; i < num_iterations / num_threads; i++) {
            result += networks.at(0)->forward(input);
        }
        local_watcher.lap();
        std::cout << torch::sum(result) << std::endl;
        std::cout << "Each thread elapse " << local_watcher.seconds() << std::endl;
    }

    void parallel_inference_thread_v2(int index) {
        torch::Tensor result = torch::zeros({batch_size, output_dim}).to(device);
        torch::Tensor input = torch::ones({batch_size, input_dim}).to(device);
        rlu::watcher::StopWatcher local_watcher;
        local_watcher.reset();
        local_watcher.start();
        for (int64_t i = 0; i < num_iterations / num_threads; i++) {
            result += networks.at(index)->forward(input);
        }
        local_watcher.lap();
        std::cout << torch::sum(result) << std::endl;
        std::cout << "Each thread elapse " << local_watcher.seconds() << std::endl;
    }

    rlu::watcher::StopWatcher watcher;
    int64_t input_dim = 11;
    int64_t num_threads = 4;
    int64_t batch_size = 64;
    int64_t num_iterations = 100000;
    int64_t output_dim = 3;
    int64_t mlp_hidden = 64;
    torch::Device device = torch::kCPU;
    std::vector<rlu::nn::StackSequential> networks;
};


TEST(parallel_torch, parallel_inference) {
    torch::manual_seed(1);
    BenchmarkParallelInference benchmark;
    benchmark.run_parallel();
    benchmark.run_serial();
    benchmark.run_parallel_network();
}

TEST(parallel_torch, parallel_gradients) {

}