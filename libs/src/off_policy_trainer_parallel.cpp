//
// Created by Chi Zhang on 8/7/21.
//

#include "trainer/off_policy_trainer_parallel.h"
#include "fmt/ostream.h"

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
    num_gradient_steps = 0;
    current_actor_index = 0;
    current_learning_index = 0;
    rlu::functional::hard_update(*actor, *agent);
}

size_t rlu::trainer::OffPolicyTrainerParallel::get_num_learners() const {
    return learner_threads.size();
}

void rlu::trainer::OffPolicyTrainerParallel::setup_environment() {
    OffPolicyTrainer::setup_environment();
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
    spdlog::info("Finish training");
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

void rlu::trainer::OffPolicyTrainerParallel::actor_fn_internal() {
    // obtain an index
    int64_t index;
    pthread_mutex_lock(&actor_index_mutex);
    index = current_actor_index;
    current_actor_index += 1;
    pthread_mutex_unlock(&actor_index_mutex);

    spdlog::info("Running actor thread {}", index);
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
            // perform logging
            int64_t epoch = global_steps_temp / steps_per_epoch;
            logger->log_tabular("Epoch", epoch);
            logger->log_tabular("EpRet", std::nullopt, true);
            logger->log_tabular("EpLen", std::nullopt, false, true);
            logger->log_tabular("TotalEnvInteracts", (float) (epoch * steps_per_epoch));
            logger->log_tabular("GradientSteps", (float) num_gradient_steps);

            // add this line if the learning is implemented.
            agent->log_tabular();

            // create a new agent. Copy the weights from the current learner
            if (online_test) {
                std::shared_ptr<agent::OffPolicyAgent> test_actor = agent_fn();
                pthread_mutex_lock(&test_actor_mutex);
                rlu::functional::hard_update(*test_actor, *actor);
                pthread_mutex_unlock(&test_actor_mutex);

                // test the current policy
                for (int i = 0; i < num_test_episodes; ++i) {
                    test_step(test_actor);
                }
            }

            // logging
            watcher.lap();

            if (online_test) {
                logger->log_tabular("TestEpRet", std::nullopt, true);
                logger->log_tabular("TestEpLen", std::nullopt, false, true);
            }

            logger->log_tabular("Time", (float) watcher.seconds());
            logger->dump_tabular();
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
            action = actor->act_single(current_obs.to(device), true).to(cpu);
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


        str_to_tensor single_data{{"obs",      current_obs},
                                  {"act",      action},
                                  {"next_obs", s.observation},
                                  {"rew",      reward_tensor},
                                  {"done",     done_tensor}};
        // store the data in a temporary buffer and wait for every batch size to store together
        // step 1: secure an index. If can't, wait in the conditional variable
        pthread_mutex_lock(&temp_buffer_mutex);
        this->temp_buffer->add_single(single_data);
        spdlog::debug("Size of the temporary buffer {}", this->temp_buffer->size());
        spdlog::debug("Size of the buffer {}", this->buffer->size());

        if (this->temp_buffer->full()) {
            // if the temporary buffer is full, compute the priority and set
            auto storage = this->temp_buffer->get_storage();
            // compute the priority
            auto priority = this->agent->compute_priority(storage.at("obs"),
                                                          storage.at("act"),
                                                          storage.at("next_obs"),
                                                          storage.at("rew"),
                                                          storage.at("done"));
            storage["priority"] = priority;
            spdlog::debug("Size of the buffer {}", this->buffer->size());
            // store data to the replay buffer
            pthread_mutex_lock(&buffer_mutex);
            buffer->add_batch(storage);
            pthread_mutex_unlock(&buffer_mutex);
            spdlog::debug("Size of the buffer {}", this->buffer->size());
            this->temp_buffer->reset();
        }
        pthread_mutex_unlock(&temp_buffer_mutex);

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
    spdlog::info("Finish actor thread {}", index);
}

void rlu::trainer::OffPolicyTrainerParallel::learner_fn_internal() {
    int64_t index;
    pthread_mutex_lock(&learning_index_mutex);
    index = current_learning_index;
    current_learning_index += 1;
    pthread_mutex_unlock(&learning_index_mutex);

    spdlog::info("Running learner thread {}", index);
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
            // TODO: when the replay buffer reaches the capacity, we need to make sure the idx is not written by the new data. Or the new priority doesn't have to update

            pthread_mutex_lock(&buffer_mutex);
            auto idx = buffer->generate_idx();
            // retrieve the actual data
            auto data = buffer->operator[](idx);
            pthread_mutex_unlock(&buffer_mutex);
            // compute the gradients
            spdlog::debug("Before compute grad");

            std::optional<torch::Tensor> importance_weights;
            if (data.contains("weights")) {
                importance_weights = data["weights"].to(device);
                auto tensorIsNan = at::isnan(importance_weights.value()).any().item<bool>();
                if (tensorIsNan) {
                    throw std::runtime_error(fmt::format("Importance weights contain NaN. {}",
                                                         importance_weights.value()));
                }
                spdlog::debug("importance weights min {}, max {}",
                              torch::min(importance_weights.value()).item<float>(),
                              torch::max(importance_weights.value()).item<float>());
            }

            auto output = agent->compute_grad(data["obs"].to(device),
                                              data["act"].to(device),
                                              data["next_obs"].to(device),
                                              data["rew"].to(device),
                                              data["done"].to(device),
                                              importance_weights,
                                              update_target);
            auto local_grads = output.first;
            output.second["idx"] = idx;
            pthread_mutex_lock(&buffer_mutex);
            this->buffer->post_process(output.second);
            pthread_mutex_unlock(&buffer_mutex);
            // update
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
            num_gradient_steps += 1;

            pthread_mutex_unlock(&learner_barrier);
        }
    }
    spdlog::info("Finish learner thread {}", index);
}

void rlu::trainer::OffPolicyTrainerParallel::tester_fn_internal(int64_t epoch) {

}

void rlu::trainer::OffPolicyTrainerParallel::start_tester_thread(int64_t epoch) {
    auto param = std::make_shared<std::pair<OffPolicyTrainerParallel *, int64_t>>(this, epoch);
    int ret = pthread_create(&tester_thread, nullptr, &tester_fn, param.get());
    if (ret != 0) spdlog::error("Fail to create actor thread with error code {}", ret);
}

void rlu::trainer::OffPolicyTrainerParallel::start_actor_threads() {
    for (auto &actor_thread: actor_threads) {
        int ret = pthread_create(&actor_thread, nullptr, &actor_fn, this);
        if (ret != 0) spdlog::error("Fail to create actor thread with error code {}", ret);
    }
}

void rlu::trainer::OffPolicyTrainerParallel::start_learner_threads() {
    for (auto &learner_thread: learner_threads) {
        int ret = pthread_create(&learner_thread, nullptr, &learner_fn, this);
        if (ret != 0) spdlog::error("Fail to create actor thread with error code {}", ret);
    }
}

void *rlu::trainer::OffPolicyTrainerParallel::actor_fn(void *param_) {
    auto *This = static_cast<OffPolicyTrainerParallel *>(param_);
    This->actor_fn_internal();
    return nullptr;
}

void *rlu::trainer::OffPolicyTrainerParallel::learner_fn(void *param_) {
    auto *This = static_cast<OffPolicyTrainerParallel *>(param_);
    This->learner_fn_internal();
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

rlu::trainer::OffPolicyTrainerParallel::~OffPolicyTrainerParallel() {
    for (auto &env: envs) {
        env->close();
    }
}

void rlu::trainer::OffPolicyTrainerParallel::setup_replay_buffer(int64_t replay_size, int64_t batch_size) {
    // TODO: add multiple temporary buffer to reduce the wait time
    OffPolicyTrainer::setup_replay_buffer(replay_size, batch_size);
}
