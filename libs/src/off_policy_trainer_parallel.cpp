//
// Created by Chi Zhang on 8/7/21.
//

#include "trainer/off_policy_trainer_parallel.h"

rlu::trainer::OffPolicyTrainerParallel::OffPolicyTrainerParallel(
        const std::function<std::shared_ptr<Gym::Environment>()> &env_fn,
        const std::function<std::shared_ptr<agent::OffPolicyAgent>()> &agent_fn, int64_t epochs,
        int64_t steps_per_epoch, int64_t start_steps, int64_t update_after, int64_t update_every,
        int64_t update_per_step, int64_t policy_delay, int64_t num_test_episodes, torch::Device device, int64_t seed,
        int64_t num_actors, int64_t num_learners) :
        OffPolicyTrainer(
                env_fn,
                agent_fn,
                epochs,
                steps_per_epoch,
                start_steps,
                update_after,
                update_every,
                update_per_step,
                policy_delay,
                num_test_episodes,
                device,
                seed
        ),
        actor(agent_fn()) {
    for (int i = 0; i < num_actors; i++) {
        actor_mutexes.emplace_back();
        actor_threads.emplace_back();
    }
    for (int i = 0; i < num_learners; i++) {
        learner_threads.emplace_back();
        grads.push_back(nullptr);
    }
    num_finished_learners = 0;
    rlu::functional::hard_update(*actor, *agent);
}

size_t rlu::trainer::OffPolicyTrainerParallel::get_num_learners() const {
    return learner_threads.size();
}

void rlu::trainer::OffPolicyTrainerParallel::setup_environment() {
    this->test_env = env_fn();
    for (size_t i = 0; i < actor_mutexes.size(); i++) {
        envs.push_back(env_fn());
    }
}

void rlu::trainer::OffPolicyTrainerParallel::train() {
    // initialize
    torch::manual_seed(seed);
    watcher.start();
    // start actor threads
    spdlog::info("Start training");
    this->start_actor_threads();
    // start learner threads
    this->start_learner_threads();
    // wait and join
    for (auto &thread: actor_threads) {
        pthread_join(thread, nullptr);
    }
    for (auto &thread: learner_threads) {
        pthread_join(thread, nullptr);
    }
}

int64_t rlu::trainer::OffPolicyTrainerParallel::get_global_steps(bool increment) {
    int64_t global_steps_temp;
    pthread_mutex_lock(&global_steps_mutex);
    if (increment) {
        total_steps += 1;
    }
    global_steps_temp = total_steps;
    pthread_mutex_unlock(&global_steps_mutex);
    return global_steps_temp;
}

void rlu::trainer::OffPolicyTrainerParallel::actor_fn_internal(size_t index) {
    spdlog::debug("Running actor thread {}", fmt::ptr(pthread_self()));
    // get environment
    auto curr_env = envs.at(index);

    Gym::State s;
    curr_env->reset(&s);
    int64_t max_global_steps = epochs * steps_per_epoch;
    // logging
    float episode_rewards = 0;
    float episode_length = 0;
    // main loop
    while (true) {
        // compute action
        torch::Tensor action;
        // copy observation
        auto current_obs = s.observation;

        int64_t global_steps_temp = this->get_global_steps(true);

        spdlog::debug("Current global step {}", global_steps_temp);
        // create a new thread for testing and logging
        if (global_steps_temp % steps_per_epoch == 0) {
            this->start_tester_thread(global_steps_temp / steps_per_epoch);
        }
        // determine whether to break
        if (global_steps_temp >= max_global_steps) {
            break;
        }


        if (global_steps_temp < start_steps) {
            spdlog::debug("Sample random actions");
            action = curr_env->action_space()->sample();
        } else {
            spdlog::debug("Start using agent action");
            // hold actor mutex. Should be mutually exclusive when copying the weights from learner to the actor
            pthread_mutex_lock(&actor_mutexes[index]);
            spdlog::debug("holding actor mutex");
            // inference on CPU
            action = actor->act_single(current_obs, true);
            spdlog::debug("After agent inference");
            pthread_mutex_unlock(&actor_mutexes[index]);
            spdlog::debug("release actor mutex");
        }

        spdlog::debug("Before step");
        // environment step
        curr_env->step(action, false, &s);
        spdlog::debug("After step");

        // TODO: need to see if it is true done or done due to reaching the maximum length.
        // convert data type
        auto reward_tensor = torch::tensor({s.reward});
        bool true_done = s.done & (!s.timeout);
        auto done_tensor = torch::tensor({true_done},
                                         torch::TensorOptions().dtype(torch::kFloat32));

        // store the data in a temporary buffer and wait for every batch size to store together
        // step 1: secure an index. If can't, wait in the conditional variable

        // step 2: add data to the index without lock

        // step 3: if the temporary buffer is full, compute priority and add to the replay buffer

        // compute the priority

        // store data to the replay buffer
        buffer->add_single({
                                   {"obs",      current_obs},
                                   {"act",      action},
                                   {"next_obs", s.observation},
                                   {"rew",      reward_tensor},
                                   {"done",     done_tensor}
                           });

        episode_rewards += s.reward;
        episode_length += 1;
        // handle terminal case
        if (s.done) {
            logger->store({
                                  {"EpRet", std::vector<float>{episode_rewards}},
                                  {"EpLen", std::vector<float>{(float) episode_length}}
                          });

            curr_env->reset(&s);
            episode_rewards = 0.;
            episode_length = 0;
        }
    }
}

void rlu::trainer::OffPolicyTrainerParallel::learner_fn_internal(size_t index) {
    spdlog::debug("Running learner thread {}", fmt::ptr(pthread_self()));
    int64_t max_global_steps = epochs * steps_per_epoch;
    while (true) {
        // get global steps
        int64_t global_steps_temp = this->get_global_steps(false);
        if (global_steps_temp >= max_global_steps) {
            break;
        }
        // TODO: use condition variable to release the CPU before update_after
        if (global_steps_temp >= update_after) {
            bool update_target = num_updates % policy_delay == 0;
            // sample a batch of data.
            // generate index
            auto idx = buffer->generate_idx();
            // retrieve the actual data
            auto data = buffer->operator[](idx);
            // compute the gradients
            spdlog::debug("Before compute grad");
            auto local_grads = agent->compute_grad(data["obs"].to(device),
                                                   data["act"].to(device),
                                                   data["next_obs"].to(device),
                                                   data["rew"].to(device),
                                                   data["done"].to(device),
                                                   std::nullopt,
                                                   update_target);
            spdlog::debug("After compute grad");
            // store the local grad in the shared memory
            grads.at(index) = std::make_shared<str_to_tensor_list>(local_grads);

            // atomically increase the done flag
            pthread_mutex_lock(&learner_barrier);
            num_finished_learners += 1;
            if (num_finished_learners == this->get_num_learners()) {
                // last thread aggregate the gradients, otherwise, conditional wait for the done signal
                auto aggregated_grads = this->aggregate_grads();
                // set the gradients
                agent->set_grad(aggregated_grads);
                // optimizer step
                agent->update_step(update_target);
                num_updates += 1;
                // atomically copy weights to actors and test_actors
                for (auto &m: actor_mutexes) {
                    pthread_mutex_lock(&m);
                }
                // TODO: need to optimize for PCIe transfer. First transfer to a CPU agent.
                // Then, perform atomic weight copy
                pthread_mutex_lock(&test_actor_mutex);
                rlu::functional::hard_update(*actor, *agent);
                pthread_mutex_unlock(&test_actor_mutex);

                for (auto &m: actor_mutexes) {
                    pthread_mutex_unlock(&m);
                }
                num_finished_learners = 0;
                // broadcast
                pthread_cond_broadcast(&learner_cond);
            } else {
                // wait the aggregator
                pthread_cond_wait(&learner_cond, &learner_barrier);
            }

            pthread_mutex_unlock(&learner_barrier);
        }

    }
}

void rlu::trainer::OffPolicyTrainerParallel::tester_fn_internal(int64_t epoch) {
    // perform logging
    logger->log_tabular("Epoch", epoch);
    logger->log_tabular("EpRet", std::nullopt, true);
    logger->log_tabular("EpLen", std::nullopt, false, true);
    logger->log_tabular("TotalEnvInteracts", (float) (epoch * steps_per_epoch));

    // add this line if the learning is implemented.
//            agent->log_tabular();

    // create a new agent. Copy the weights from the current learner
    std::shared_ptr<agent::OffPolicyAgent> test_actor = agent_fn();
    pthread_mutex_lock(&test_actor_mutex);
    rlu::functional::hard_update(*test_actor, *actor);
    pthread_mutex_unlock(&test_actor_mutex);

    // test the current policy
    for (int i = 0; i < num_test_episodes; ++i) {
        test_step(test_actor);
    }
    // logging
    watcher.lap();

    logger->log_tabular("TestEpRet", std::nullopt, true);
    logger->log_tabular("TestEpLen", std::nullopt, false, true);
    logger->log_tabular("Time", (float) watcher.seconds());

    logger->dump_tabular();
}

void rlu::trainer::OffPolicyTrainerParallel::start_tester_thread(int64_t epoch) {
    auto param = std::make_shared<std::pair<OffPolicyTrainerParallel *, int64_t>>(this, epoch);
    int ret = pthread_create(&tester_thread, nullptr, &tester_fn, param.get());
    if (ret != 0) spdlog::error("Fail to create actor thread with error code {}", ret);
}

void rlu::trainer::OffPolicyTrainerParallel::start_actor_threads() {
    for (size_t i = 0; i < actor_threads.size(); i++) {
        auto param = std::make_shared<std::pair<OffPolicyTrainerParallel *, size_t>>(this, i);
        int ret = pthread_create(&actor_threads[i], nullptr, &actor_fn, param.get());
        if (ret != 0) spdlog::error("Fail to create actor thread with error code {}", ret);
    }
}

void rlu::trainer::OffPolicyTrainerParallel::start_learner_threads() {
    for (size_t i = 0; i < learner_threads.size(); i++) {
        auto param = std::make_shared<std::pair<OffPolicyTrainerParallel *, size_t>>(this, i);
        int ret = pthread_create(&learner_threads[i], nullptr, &learner_fn, param.get());
        if (ret != 0) spdlog::error("Fail to create actor thread with error code {}", ret);
    }
}

void *rlu::trainer::OffPolicyTrainerParallel::actor_fn(void *param_) {
    auto param = (std::pair<OffPolicyTrainerParallel *, size_t> *) param_;
    OffPolicyTrainerParallel *This = param->first;
    size_t index = param->second;
    This->actor_fn_internal(index);
    return nullptr;
}

void *rlu::trainer::OffPolicyTrainerParallel::learner_fn(void *param_) {
    auto param = (std::pair<OffPolicyTrainerParallel *, size_t> *) param_;
    OffPolicyTrainerParallel *This = param->first;
    size_t index = param->second;
    This->learner_fn_internal(index);
    return nullptr;
}

void *rlu::trainer::OffPolicyTrainerParallel::tester_fn(void *param_) {
    auto param = (std::pair<OffPolicyTrainerParallel *, int64_t> *) param_;
    OffPolicyTrainerParallel *This = param->first;
    This->tester_fn_internal(param->second);
    return nullptr;
}

rlu::str_to_tensor_list rlu::trainer::OffPolicyTrainerParallel::aggregate_grads() {
    str_to_tensor_list result;
    for (auto &it: *grads.at(0)) {
        result[it.first] = torch::autograd::variable_list(it.second.size());
        // for each parameter
        for (size_t i = 0; i < it.second.size(); i++) {
            std::vector<torch::Tensor> current_grad;
            // for each learner
            for (auto &learner_grad: grads) {
                auto param_grad = learner_grad->at(it.first); // a list of tensor
                current_grad.push_back(param_grad.at(i));
            }
            // aggregate
            result[it.first].at(i) = torch::mean(torch::stack(current_grad, 0), 0);
        }
    }
    return result;
}
