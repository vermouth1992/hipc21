//
// Created by chi on 10/27/21.
//

/*
 * Run num_actors parallel thread to insert into the replay buffer and num_learners thread to sample from the replay
 * buffer. Use sleep to represent the training and inference. Compare global synchronization vs. optimized
 */

#include "gtest/gtest.h"
#include "replay_buffer/replay_buffer.h"

#include <utility>
#include <thread>
#include <mutex>
#include <chrono>
#include "utils/stop_watcher.h"

class BenchmarkReplayBuffer {
public:
    explicit BenchmarkReplayBuffer(int num_actors, int num_learners,
                                   int training_time, int inference_time,
                                   std::shared_ptr<rlu::replay_buffer::ReplayBuffer> buffer) :
            training_time(training_time), inference_time(inference_time),
            num_actors(num_actors), num_learners(num_learners), buffer(std::move(buffer)) {
        actor_idx = 0;
        learner_idx = 0;
        inference_steps = 0;
        learning_steps = 0;
    }

    void run() {
        for (int i = 0; i < num_actors; i++) {
            actor_threads.emplace_back(&BenchmarkReplayBuffer::actor_fn, this);
        }
        for (int j = 0; j < num_learners; j++) {
            learner_threads.emplace_back(&BenchmarkReplayBuffer::learner_fn, this);
        }
        for (auto &t: actor_threads) {
            t.join();
        }
        for (auto &t: learner_threads) {
            t.join();
        }
    }

    [[nodiscard]] rlu::str_to_tensor inference() const {
        std::this_thread::sleep_for(std::chrono::milliseconds(inference_time));
        // generate random data
        return {}
    }

    void actor_fn() {
        int index = atomic_increase_get_index(actor_idx_mutex, actor_idx);
        std::cout << "Actor " << index << " running" << std::endl;

        while (true) {
            // perform inference
            auto data = inference();
            // insert into the replay buffer as a batch
            buffer->add_batch(data);
            // increment the global iteration index


            // wait for the learner if it is too fast
        }
    }

    void learner_fn() {
        int index = atomic_increase_get_index(learner_idx_mutex, learner_idx);
        std::cout << "Learner " << index << " running" << std::endl;

        // sample data

        // perform learning

        // update the priority

    }

private:
    static int atomic_increase_get_index(std::mutex &mutex, int &index) {
        // atomically increase the index and return
        int temp;
        mutex.lock();
        temp = index;
        index += 1;
        mutex.unlock();
        return temp;
    }

    // global variables to assign thread index
    int actor_idx;
    int learner_idx;
    std::mutex actor_idx_mutex;
    std::mutex learner_idx_mutex;
    // global variables to measure the training progress
    int inference_steps;
    int learning_steps;
    std::mutex inference_steps_mutex;
    std::mutex learning_steps_mutex;
    // threads
    std::vector<std::thread> actor_threads;
    std::vector<std::thread> learner_threads;

    // parameters
    int training_time;
    int inference_time;
    int num_actors;
    int num_learners;
    std::shared_ptr<rlu::replay_buffer::ReplayBuffer> buffer;
};


TEST(replay_buffer, synchronization) {
    int replay_size = 1000000;
    int batch_size = 100;
    // Hopper-v2
    rlu::str_to_dataspec data_spec{
            {"obs",      rlu::DataSpec({11}, torch::kFloat32)},
            {"act",      rlu::DataSpec({3}, torch::kFloat32)},
            {"next_obs", rlu::DataSpec({11}, torch::kFloat32)},
            {"rew",      rlu::DataSpec({}, torch::kFloat32)},
            {"done",     rlu::DataSpec({}, torch::kFloat32)}
    };
    std::shared_ptr<rlu::replay_buffer::ReplayBuffer> buffer = std::make_shared<rlu::replay_buffer::PrioritizedReplayBuffer<rlu::replay_buffer::SegmentTreeNary>>(
            replay_size, data_spec, batch_size, 0.6);
    BenchmarkReplayBuffer b(5, 4, 100, 100, buffer);
    b.run();
}