//
// Created by chi on 10/27/21.
//

/*
 * Run num_actors parallel thread to insert into the replay buffer and num_learners thread to sample from the replay
 * buffer. Use sleep to represent the training and inference. Compare global synchronization vs. optimized
 */

#include "gtest/gtest.h"

#include "replay_buffer/prioritized_replay_buffer.h"
#include "replay_buffer/prioritized_replay_buffer_global.h"
#include "replay_buffer/prioritized_replay_buffer_optimized.h"

#include <fstream>
#include <thread>
#include <mutex>
#include <chrono>
#include "utils/stop_watcher.h"

#define obs_dim 11
#define act_dim 3


class BenchmarkReplayBuffer {
public:
    explicit BenchmarkReplayBuffer(int num_actors, int num_learners,
                                   int training_time, int inference_time) :
            training_time(training_time), inference_time(inference_time),
            num_actors(num_actors), num_learners(num_learners) {
        actor_idx = 0;
        learner_idx = 0;
        inference_steps = 0;
        learning_steps = 0;
    }

    void set_buffer(std::shared_ptr<rlu::replay_buffer::ReplayBuffer> buffer_) {
        this->buffer = buffer_;
    }

    double run() {
        rlu::watcher::StopWatcher timer;
        timer.start();
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
        timer.lap();
        std::cout << "Time elapsed " << timer.seconds() << std::endl;
        return timer.seconds();
    }

    [[nodiscard]] rlu::str_to_tensor inference() const {
        std::this_thread::sleep_for(std::chrono::milliseconds(inference_time));
        // generate random data
        return {
                {"obs",      torch::randn({1, obs_dim}, torch::TensorOptions().dtype(torch::kFloat32))},
                {"act",      torch::randn({1, act_dim}, torch::TensorOptions().dtype(torch::kFloat32))},
                {"next_obs", torch::randn({1, obs_dim}, torch::TensorOptions().dtype(torch::kFloat32))},
                {"rew",      torch::randn({1}, torch::TensorOptions().dtype(torch::kFloat32))},
                {"done",     torch::randn({1}, torch::TensorOptions().dtype(torch::kFloat32))},
        };
    }

    [[nodiscard]] rlu::str_to_tensor learn(const rlu::str_to_tensor &data) const {
        auto batch_size = data.at("obs").sizes().at(0);
        std::this_thread::sleep_for(std::chrono::milliseconds(training_time));
        return {
                {"priority", torch::randn(batch_size, torch::TensorOptions().dtype(torch::kFloat32))}
        };
    }

    void actor_fn() {
        int index = atomic_increment_get(actor_idx_mutex, actor_idx);
        std::cout << "Actor " << index << " running" << std::endl;

        while (true) {
            // increment the global iteration index
            auto global_steps = atomic_increment_get(inference_steps_mutex, inference_steps);
            cv_update_after.notify_all();

//            std::cout << "Global step " << global_steps << std::endl;
            if (global_steps > total_steps) break;

            // perform inference
            auto data = inference();
            // insert into the replay buffer as a batch
            buffer->add_batch(data);
            // wait for the learner if it is too fast
            {
                // hold mutex
                std::unique_lock<std::mutex> lk(learning_steps_mutex);
                cv.wait(lk, [&global_steps, this] { return (global_steps - learning_steps) < 1000; });
            }
        }
        std::cout << "Actor " << index << " finishes" << std::endl;
    }

    void learner_fn() {
        int learner_index = atomic_increment_get(learner_idx_mutex, learner_idx);
        std::cout << "Learner " << learner_index << " running" << std::endl;

        // wait for update_after
        {
            std::unique_lock<std::mutex> lk(inference_steps_mutex);
            cv_update_after.wait(lk, [this] { return inference_steps > update_after; });
        }

        while (true) {
            // increment and notify
            auto global_steps = atomic_increment_get(learning_steps_mutex, learning_steps);
            cv.notify_all();
            if (global_steps > total_steps) break;
            // sample data
            auto data = this->buffer->sample(0);
            // perform learning
            auto log = this->learn(data);
            log["idx"] = data["idx"];
            // update the priority
            this->buffer->post_process(log);

        }
        std::cout << "Learner " << learner_index << " finishes" << std::endl;
    }

private:
    static int atomic_increment_get(std::mutex &mutex, int &index, bool increment = true) {
        // atomically increase the index and return
        int temp;
        mutex.lock();
        temp = index;
        if (increment) {
            index += 1;
        }
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
    // synchronization variables
    std::condition_variable cv;
    std::condition_variable cv_update_after;
    // const variables
    const int total_steps = 5000;
    const int update_after = 1000;
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
    int training_time = 0;
    int inference_time = 0;
    // Hopper-v2
    std::vector<int> num_actors_lst{4, 8, 12, 16, 20};
    // output file
    std::ofstream output_file;
    double result;
    output_file.open("replay_sync.csv");
    output_file << "type,num_actors,num_learners,time\n";

    for (auto &num_actors: num_actors_lst) {
        int num_learners = num_actors / 2;
        rlu::str_to_dataspec data_spec{
                {"obs",      rlu::DataSpec({obs_dim}, torch::kFloat32)},
                {"act",      rlu::DataSpec({act_dim}, torch::kFloat32)},
                {"next_obs", rlu::DataSpec({obs_dim}, torch::kFloat32)},
                {"rew",      rlu::DataSpec({}, torch::kFloat32)},
                {"done",     rlu::DataSpec({}, torch::kFloat32)}
        };
        std::shared_ptr<rlu::replay_buffer::ReplayBuffer> buffer = std::make_shared<rlu::replay_buffer::PrioritizedReplayBufferGlobalLock<rlu::replay_buffer::SegmentTreeNary>>(
                replay_size, data_spec, batch_size, 0.6);
        BenchmarkReplayBuffer b(num_actors, num_learners, training_time, inference_time);
        b.set_buffer(buffer);
        result = b.run();
        output_file << "Global," << num_actors << "," << num_learners << "," << result << std::endl;

        std::shared_ptr<rlu::replay_buffer::ReplayBuffer> buffer_no_sync = std::make_shared<rlu::replay_buffer::PrioritizedReplayBuffer<rlu::replay_buffer::SegmentTreeNary>>(
                replay_size, data_spec, batch_size, 0.6
        );
        BenchmarkReplayBuffer b1(num_actors, num_learners, training_time, inference_time);
        b1.set_buffer(buffer_no_sync);
        result = b1.run();

        output_file << "NoLock," << num_actors << "," << num_learners << "," << result << std::endl;

        std::shared_ptr<rlu::replay_buffer::ReplayBuffer> buffer_sync_opt = std::make_shared<rlu::replay_buffer::PrioritizedReplayBufferOpt<rlu::replay_buffer::SegmentTreeNary>>(
                replay_size, data_spec, batch_size, 0.6, num_actors + num_learners
        );
        BenchmarkReplayBuffer b2(num_actors, num_learners, training_time, inference_time);
        b2.set_buffer(buffer_sync_opt);
        result = b2.run();

        output_file << "Optimized," << num_actors << "," << num_learners << "," << result << std::endl;
    }


}