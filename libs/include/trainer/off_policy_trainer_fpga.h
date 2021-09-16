//
// Created by Chi Zhang on 9/8/21.
//

#ifndef HIPC21_OFF_POLICY_TRAINER_FPGA_H
#define HIPC21_OFF_POLICY_TRAINER_FPGA_H


/*
 * The steps for training on FPGA
 */

#include "off_policy_trainer_parallel.h"

namespace rlu::trainer {
    class OffPolicyTrainerFPGA : public OffPolicyTrainerParallel {
    public:
        explicit OffPolicyTrainerFPGA(const std::function<std::shared_ptr<Gym::Environment>()> &env_fn,
                                      const std::function<std::shared_ptr<agent::OffPolicyAgent>()> &agent_fn,
                                      int64_t epochs,
                                      int64_t steps_per_epoch,
                                      int64_t start_steps,
                                      int64_t update_after,
                                      int64_t update_every,
                                      int64_t update_per_step,
                                      int64_t policy_delay,
                                      int64_t num_test_episodes,
                                      int64_t seed,
                                      int64_t num_actors,
                                      const std::string &bitstream_path) :
                OffPolicyTrainerParallel(
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
                        torch::kCPU,
                        seed,
                        num_actors,
                        1
                ) {
            this->initialize_bitstream(bitstream_path);
        }

        void setup_replay_buffer(int64_t replay_size, int64_t batch_size) override {
            // must using the FPGA based replay buffer, where the subtree is on the FPGA and the data is stored
            // in the CPU memory
            std::unique_ptr<DataSpec> action_data_spec;
            auto action_space = env->action_space();
            auto observation_space = env->observation_space();
            if (action_space->type == action_space->DISCRETE) {
                action_data_spec = std::make_unique<DataSpec>(std::vector<int64_t>(), torch::kInt64);
            } else {
                action_data_spec = std::make_unique<DataSpec>(action_space->box_shape, torch::kFloat32);
            }
            // setup agent
            agent->to(device);
            // setup replay buffer
            str_to_dataspec data_spec{
                    {"obs",      DataSpec(observation_space->box_shape, torch::kFloat32)},
                    {"act",      *action_data_spec},
                    {"next_obs", DataSpec(observation_space->box_shape, torch::kFloat32)},
                    {"rew",      DataSpec({}, torch::kFloat32)},
                    {"done",     DataSpec({}, torch::kFloat32)},
            };

            this->buffer = std::make_shared<rlu::replay_buffer::PrioritizedReplayBuffer>(
                    replay_size, data_spec, batch_size, 0.8, "fpga");
        }


    protected:
        void learner_fn_internal(size_t index) override {
            // we assume there is only one learner here.
            int64_t max_global_steps = epochs * steps_per_epoch;
            while (true) {
                // get global steps
                int64_t global_steps_temp = this->get_global_steps(false);
                if (global_steps_temp >= max_global_steps) {
                    break;
                }

                // step 1: query FPGA about idx
//                auto idx = buffer->generate_idx();
                // retrieve the actual data
//                auto data = buffer->operator[](idx);
                // send the data to the FPGA and waits for the FPGA to complete and send back logging data including
                // the QVals (batch) and the loss of Q (scalar)

                // increase the number of gradient steps

                // copy the weights from FPGA to the CPU
                synchronize_weights();
            }


        }

    private:
        // initialize the bitstream
        void initialize_bitstream(const std::string &filepath) {
            // TODO: initialize the bitstream and push to the FPGA
        }

        void synchronize_weights() {
            // synchronize the weights from the FPGA to the CPU. The CPU weights is in
            // actor->parameters(). Actor is a torch::nn::Module. It is a 3-layer MLP with relu activation
            // The weights can be referred by index. For example, actor->parameters()[0], etc
        }

    };

}


#endif //HIPC21_OFF_POLICY_TRAINER_FPGA_H
