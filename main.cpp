//
// Created by chi on 7/2/21.
//

#include "include/gym/gym.h"
#include "dqn.h"
#include <iostream>

int main(int argc, char **argv) {
    int epochs = std::stoi(argv[1]);
    std::string env_id = argv[2];
    try {
//        torch::set_num_threads(1);
        std::cout << torch::get_num_threads() << " " << torch::get_thread_num() << std::endl;
        torch::manual_seed(1);

        torch::DeviceType device_type = torch::kCPU;
//        if (torch::cuda::is_available()) {
//            std::cout << "CUDA available! Training on GPU." << std::endl;
//            device_type = torch::kCUDA;
//        } else {
//            std::cout << "Training on CPU." << std::endl;
//            device_type = torch::kCPU;
//        }
        torch::Device device(device_type);
        boost::shared_ptr<Gym::Client> client = Gym::client_create("127.0.0.1", 5000);
        train_dqn(client, env_id, epochs, 1000, 1000,
                  500, 1, 1, 1, 100,
                  10, 1, 1000000, 64, true, 0.99, 1e-3,
                  5e-3, 0.1, device);

    } catch (const std::exception &e) {
        fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }

    return 0;
}