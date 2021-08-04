//
// Created by Chi Zhang on 8/2/21.
//

#include "gtest/gtest.h"
#include "gym/gym.h"
#include <iostream>

static
void run_single_environment(
        const std::shared_ptr<Gym::Client> &client,
        const std::string &env_id,
        int episodes_to_run) {
    std::shared_ptr<Gym::Environment> env = client->make(env_id);
    std::shared_ptr<Gym::Space> action_space = env->action_space();
    std::shared_ptr<Gym::Space> observation_space = env->observation_space();

    for (int e = 0; e < episodes_to_run; ++e) {
        printf("%s episode %i...\n", env_id.c_str(), e);
        Gym::State s;
        env->reset(&s);
        std::cout << s.observation.sizes() << std::endl;
        float total_reward = 0;
        int total_steps = 0;
        while (true) {
            auto action = action_space->sample();
            env->step(action, false, &s);
            total_reward += s.reward;
            total_steps += 1;
            if (s.done) break;
        }
        printf("%s episode %i finished in %i steps with reward %0.2f\n",
               env_id.c_str(), e, total_steps, total_reward);
    }
    env->close();
}

TEST(env, basic) {
    try {
        std::shared_ptr<Gym::Client> client = Gym::client_create("127.0.0.1", 5000);
        run_single_environment(client, "Pong-v0", 1);

    } catch (const std::exception &e) {
        fprintf(stderr, "ERROR: %s\n", e.what());
    }
}