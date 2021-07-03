//
// Created by chi on 7/2/21.
//

#include "include/gym/gym.h"
#include "dqn.h"
#include <iostream>

int main() {
    try {
        boost::shared_ptr<Gym::Client> client = Gym::client_create("127.0.0.1", 5000);
        train_dqn(client, "CartPole-v0", 100, 1000, 1000,
                  500, 1, 1, 1, 100,
                  10, 1, 1000000, 256, true, 0.99, 1e-3,
                  5e-3, 0.1);

    } catch (const std::exception &e) {
        fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }

    return 0;
}