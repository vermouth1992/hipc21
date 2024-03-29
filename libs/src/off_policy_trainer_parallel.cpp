//
// Created by Chi Zhang on 8/7/21.
//

#include "trainer/off_policy_trainer_parallel.h"
#include "fmt/ostream.h"

rlu::trainer::OffPolicyTrainerParallel::OffPolicyTrainerParallel(
        const std::function<std::shared_ptr<gym::env::Env>()> &env_fn,
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
    agent->to(device);

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
    auto index = this->get_actor_index();
    spdlog::info("Running actor thread {}", index);
    // get environment
    auto curr_env = envs.at(index);

    gym::env::State s;
    curr_env->reset(s);
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
        spdlog::debug("Before wake up learner");
        this->wake_up_learner();

        spdlog::debug("Before get_update_steps");
        int64_t num_updates_temp = this->get_update_steps(false);

        // logging
        spdlog::debug("Before logging");
        this->log(global_steps_temp, num_updates_temp);
        spdlog::debug("After logging");

        // determine whether to break
        if (global_steps_temp >= max_global_steps) {
            break;
        }

        spdlog::debug("Current global step {}", global_steps_temp);

        // TODO: make sure the actor and the target approximately meets the update_per_step
        // If the actors are too fast, wait for the learner.
        // If the actors are slow, simply increase the number of actors
        this->actor_wait_for_learner(global_steps_temp);


        if (global_steps_temp < start_steps) {
            spdlog::debug("Sample random actions");
            action = curr_env->action_space->sample();
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
        curr_env->step(action, s);
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

        // try to acquire the lock. If failed, try to get the next one
        size_t lock_index = 0;
        spdlog::debug("before getting mutex");
        while (true) {
            int ret = pthread_mutex_trylock(&this->temp_buffer_mutex.at(lock_index));
            if (ret == 0) {
                break;
            } else {
                lock_index = (lock_index + 1) % this->temp_buffer_mutex.size();
            }
        }

        spdlog::debug("before add_single(single_data)");
        this->temp_buffer.at(lock_index)->add_single(single_data);
        spdlog::debug("add_single(single_data)");
        if (this->temp_buffer.at(lock_index)->full()) {
            spdlog::debug("Temp buffer {} is full", lock_index);
            // if the temporary buffer is full, compute the priority and set
            auto storage = this->temp_buffer.at(lock_index)->get();
            // compute the priority.
            // TODO: need mutex for inference
            spdlog::debug("Before computing priority");
            pthread_mutex_lock(&agent_mutexes.at(lock_index));
            // TODO: add segment tree GPU
            auto priority = this->agent->compute_priority(storage.at("obs").to(device),
                                                          storage.at("act").to(device),
                                                          storage.at("next_obs").to(device),
                                                          storage.at("rew").to(device),
                                                          storage.at("done").to(device)).cpu();
            pthread_mutex_unlock(&agent_mutexes.at(lock_index));
            spdlog::debug("After computing priority");
            storage["priority"] = priority;
            spdlog::debug("Size of the buffer {}", this->buffer->size());
            // store data to the replay buffer
            pthread_mutex_lock(&buffer_mutex);
            buffer->add_batch(storage);
            pthread_mutex_unlock(&buffer_mutex);
            spdlog::debug("Size of the buffer {}", this->buffer->size());
            this->temp_buffer.at(lock_index)->reset();
        }
        pthread_mutex_unlock(&this->temp_buffer_mutex.at(lock_index));

        episode_rewards += s.reward;
        episode_length += 1;
        spdlog::debug("Before done");
        // handle terminal case
        if (s.done) {
            spdlog::debug("Before store");
            logger->store({
                                  {"EpRet", std::vector<float>{episode_rewards}},
                                  {"EpLen", std::vector<float>{(float) episode_length}}
                          });
            spdlog::debug("After done");
            curr_env->reset(s);
            spdlog::debug("After reset");
            episode_rewards = 0.;
            episode_length = 0;
        }
    }

    // wait for the logger
    sleep(1);
    spdlog::info("Finish actor thread {}", index);
}

void rlu::trainer::OffPolicyTrainerParallel::learner_fn_internal() {
    auto index = this->get_learner_index();
    this->learner_wait_to_start();
    spdlog::info("Running learner thread {}", index);

    int64_t max_global_steps = epochs * steps_per_epoch;
    // start learning
    while (true) {
        // get global steps
        int64_t global_steps_temp = this->get_global_steps(false);
        if (global_steps_temp >= max_global_steps) {
            break;
        }

        bool update_target = num_updates % policy_delay == 0;
        // sample a batch of data.
        // generate index
        // TODO: when the replay buffer reaches the capacity, we need to make sure the idx is not written by the new data. Or the new priority doesn't have to update
        spdlog::debug("Before generating index");
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
        }

        auto output = agent->compute_grad(data["obs"].to(device),
                                          data["act"].to(device),
                                          data["next_obs"].to(device),
                                          data["rew"].to(device),
                                          data["done"].to(device),
                                          importance_weights,
                                          update_target);
        auto local_grads = output.first;
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


            // acquire all the inference lock
            for (auto &agent_mutex: agent_mutexes) {
                pthread_mutex_lock(&agent_mutex);
            }
            // optimizer step
            agent->update_step(update_target);
            for (auto &agent_mutex: agent_mutexes) {
                pthread_mutex_unlock(&agent_mutex);
            }

            for (auto &m: actor_mutexes) {
                pthread_mutex_lock(&m);
            }
            // TODO: need to optimize for PCIe transfer. First transfer to a CPU agent.
            pthread_mutex_lock(&test_actor_mutex);
            // copy the weight of the agent to CPU
            rlu::functional::hard_update(*actor, *agent);
            pthread_mutex_unlock(&test_actor_mutex);

            for (auto &m: actor_mutexes) {
                pthread_mutex_unlock(&m);
            }
            num_finished_learners = 0;

            this->get_update_steps(true);
            this->wake_up_actor();

            // broadcast
            pthread_cond_broadcast(&learner_cond);
        } else {
            // wait the aggregator
            pthread_cond_wait(&learner_cond, &learner_barrier);
        }

        output.second["idx"] = idx;
        pthread_mutex_lock(&buffer_mutex);
        this->buffer->post_process(output.second);
        pthread_mutex_unlock(&buffer_mutex);

        pthread_mutex_unlock(&learner_barrier);

    }
    // wait for the logger
    sleep(1);
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
    try {
        auto *This = static_cast<OffPolicyTrainerParallel *>(param_);
        This->actor_fn_internal();
    } catch (const std::exception &e) {
        fprintf(stderr, "ERROR when running the actor: %s\n", e.what());
    }
    return nullptr;
}

void *rlu::trainer::OffPolicyTrainerParallel::learner_fn(void *param_) {
    try {
        auto *This = static_cast<OffPolicyTrainerParallel *>(param_);
        This->learner_fn_internal();
    } catch (const std::exception &e) {
        fprintf(stderr, "ERROR when running the learner: %s\n", e.what());
    }
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
    batch_size = batch_size / (int64_t) this->get_num_learners();
    OffPolicyTrainer::setup_replay_buffer(replay_size, batch_size);
    // add num_actors - 1 more temp_buffer
    for (size_t i = 0; i < this->get_num_actors() - 1; i++) {
        this->temp_buffer.push_back(std::make_shared<replay_buffer::UniformReplayBuffer>(
                batch_size, this->buffer->get_data_spec(), 1));
        this->temp_buffer_mutex.emplace_back();
        this->agent_mutexes.emplace_back();
    }
    this->temp_buffer_mutex.emplace_back();
    this->agent_mutexes.emplace_back();
}

size_t rlu::trainer::OffPolicyTrainerParallel::get_num_actors() const {
    return this->actor_threads.size();
}

void rlu::trainer::OffPolicyTrainerParallel::actor_wait_for_learner(int64_t global_steps_temp) {
    pthread_mutex_lock(&update_steps_mutex);
    while (num_updates < (global_steps_temp - update_after) * update_per_step) {
        // conditional wait
        pthread_cond_wait(&update_steps_cond, &update_steps_mutex);
    }
    pthread_mutex_unlock(&update_steps_mutex);
}

void rlu::trainer::OffPolicyTrainerParallel::log(int64_t global_steps_temp, int64_t num_updates_temp) {
    // create a new thread for testing and logging
    if (global_steps_temp % steps_per_epoch == 0) {
        // perform logging
        int64_t epoch = global_steps_temp / steps_per_epoch;
        logger->log_tabular("Epoch", epoch);
        logger->log_tabular("EpRet", std::nullopt, true);
        logger->log_tabular("EpLen", std::nullopt, false, true);
        logger->log_tabular("TotalEnvInteracts", (float) global_steps_temp);
        logger->log_tabular("GradientSteps", (float) num_updates_temp);

        spdlog::debug("After standard logging");

        // add this line if the learning is implemented.
        agent->log_tabular();

        spdlog::debug("After agent logging");

        // create a new agent. Copy the weights from the current learner
        if (online_test) {
            std::shared_ptr<agent::OffPolicyAgent> test_actor = agent_fn();
            pthread_mutex_lock(&test_actor_mutex);
            rlu::functional::hard_update(*test_actor, *actor);
            pthread_mutex_unlock(&test_actor_mutex);

            // TODO: test the current policy
        }

        // logging
        watcher.lap();

        if (online_test) {
            logger->log_tabular("TestEpRet", std::nullopt, true);
            logger->log_tabular("TestEpLen", std::nullopt, false, true);
        }

        logger->log_tabular("Time", (float) watcher.seconds());
        logger->dump_tabular();

        spdlog::debug("After dump tabular");
    }
}

int64_t rlu::trainer::OffPolicyTrainerParallel::get_actor_index() {
    int64_t index;
    pthread_mutex_lock(&actor_index_mutex);
    index = current_actor_index;
    current_actor_index += 1;
    pthread_mutex_unlock(&actor_index_mutex);
    return index;
}

int64_t rlu::trainer::OffPolicyTrainerParallel::get_learner_index() {
    int64_t index;
    pthread_mutex_lock(&learning_index_mutex);
    index = current_learning_index;
    current_learning_index += 1;
    pthread_mutex_unlock(&learning_index_mutex);
    return index;
}

void rlu::trainer::OffPolicyTrainerParallel::learner_wait_to_start() {
    pthread_mutex_lock(&global_steps_mutex);
    auto global_steps_temp = total_steps;
    while (global_steps_temp < update_after) {
        global_steps_temp = total_steps;
        pthread_cond_wait(&learner_init_wait_cond, &global_steps_mutex);
    }
    pthread_mutex_unlock(&global_steps_mutex);
}

void rlu::trainer::OffPolicyTrainerParallel::wake_up_actor() {
    pthread_mutex_lock(&update_steps_mutex);
    pthread_cond_broadcast(&update_steps_cond);
    pthread_mutex_unlock(&update_steps_mutex);
}

void rlu::trainer::OffPolicyTrainerParallel::wake_up_learner() {
    // wake up the learner threads
    pthread_mutex_lock(&global_steps_mutex);
    if (total_steps >= update_after) {
        pthread_cond_broadcast(&learner_init_wait_cond);
    }
    pthread_mutex_unlock(&global_steps_mutex);
}

int64_t rlu::trainer::OffPolicyTrainerParallel::get_update_steps(bool increment) {
    pthread_mutex_lock(&update_steps_mutex);
    if (increment) {
        num_updates += 1;
    }
    int64_t temp = num_updates;
    pthread_mutex_unlock(&update_steps_mutex);
    return temp;
}
