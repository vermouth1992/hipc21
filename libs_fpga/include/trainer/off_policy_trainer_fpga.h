//
// Created by Chi Zhang on 9/8/21.
//

#ifndef HIPC21_OFF_POLICY_TRAINER_FPGA_H
#define HIPC21_OFF_POLICY_TRAINER_FPGA_H


/*
 * The steps for training on FPGA
 */

#include "trainer/off_policy_trainer_parallel.h"
#include "replay_buffer/segment_tree_fpga.h"
// #include "fmt/ostream.h"
#include "fpga.h"
#include "rmm.h"
#include "topl_new.h"

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
            auto action_space = test_env->action_space();
            auto observation_space = test_env->observation_space();
            if (action_space->type == action_space->DISCRETE) {
                action_data_spec = std::make_unique<DataSpec>(std::vector<int64_t>(), torch::kInt64);
            } else {
                action_data_spec = std::make_unique<DataSpec>(action_space->box_shape, torch::kFloat32);
            }
            // setup replay buffer
            str_to_dataspec data_spec{
                    {"obs",      DataSpec(observation_space->box_shape, torch::kFloat32)},
                    {"act",      *action_data_spec},
                    {"next_obs", DataSpec(observation_space->box_shape, torch::kFloat32)},
                    {"rew",      DataSpec({}, torch::kFloat32)},
                    {"done",     DataSpec({}, torch::kFloat32)},
            };

            this->buffer = std::make_shared<rlu::replay_buffer::PrioritizedReplayBuffer<rlu::replay_buffer::SegmentTreeFPGA>>(
                    replay_size, data_spec, batch_size, 0.6);
            for (size_t i = 0; i < this->get_num_actors(); i++) {
                this->temp_buffer.push_back(std::make_shared<replay_buffer::UniformReplayBuffer>(
                        batch_size, this->buffer->get_data_spec(), 1));
                this->temp_buffer_mutex.emplace_back();
                this->agent_mutexes.emplace_back();
            }
        }


    protected:
        void learner_fn_internal() override {
            spdlog::info("Learner waits to start");
            this->learner_wait_to_start();
            // we assume there is only one learner here.
            spdlog::info("Learner starts");
            int64_t max_global_steps = epochs * steps_per_epoch;

            cl::Kernel krnl_top(rlu::fpga::program, "learners_top:{top_1}");
            cl::Kernel krnl_tree(rlu::fpga::program, "Top_tree", &rlu::fpga::err); // Replay Update (Parallel with train):

            while (true) {
                // get global steps
                int64_t global_steps_temp = this->get_global_steps(false);
                if (global_steps_temp >= max_global_steps) {
                    break;
                }
                // if (global_steps_temp >= update_after) {
                spdlog::debug("In loop");
                // step 1: query FPGA about idx
                pthread_mutex_lock(&buffer_mutex);
                auto idx = buffer->generate_idx();
                spdlog::debug("generate_idx");

                // retrieve the actual data
                auto data = buffer->operator[](idx);
                spdlog::debug("operator[]");
                pthread_mutex_unlock(&buffer_mutex);
                // =========================send the data to the FPGA and waits for the FPGA to complete and send back logging data including
                // the QVals (batch) and the loss of Q (scalar)
                // std::cout << "Host: init input states..." << std::endl;
                // spdlog::info("Host: init input states...");
                
                // std::cout<<"Data[obs].size():"<< data["obs"].sizes() <<"\n";
                // std::cout<<"Data[obs].size():"<< data["next_obs"].sizes() <<"\n";

                // rlu::watcher::StopWatcher watcher;
                // watcher.reset();
                // watcher.start();

                for (int jj = 0; jj < BATCHS; jj++) {   
                    for (int j = 0; j < L1; j++) {
                        for (int i = 0; i < BSIZE; i++) {
                        rlu::fpga::In_rows[L1*jj+j].a[i] = data["obs"].index({(jj*BSIZE+i),j}).item<float>(); //[jj*BSIZE+i][j];   //?????????????????
                        rlu::fpga::In_rows_snt[L1*jj+j].a[i] = data["next_obs"].index({(jj*BSIZE+i),j}).item<float>();  //[jj*BSIZE+i][j];  //?????????????????
                        // printf("%f ",In_rows[j].a[i]);         
                        // printf("%f ",In_rows_snt[j].a[i]);
                        }
                    }
                }

                // printf("\nHost: Init input reward/action/done content...\n");
                spdlog::debug("Host: Init input reward/action/done content...");

                for (int jj = 0; jj < BATCHS; jj++) {   
                    for (int i = 0; i < BSIZE; i++) {
                    // printf("\njj,i:%d,%d\n",jj,i);
                        rlu::fpga::In_actions[jj].a[i] = data["act"].index({jj*BSIZE+i}).item<float>();
                        rlu::fpga::In_rewards[jj].a[i] = data["rew"].index({jj*BSIZE+i}).item<float>();
                        rlu::fpga::In_dones[jj].a[i] = data["done"].index({jj*BSIZE+i}).item<float>();
                        // printf("%d ",In_actions[jj].a[i]);
                    }
                }

                rlu::fpga::gamma = 0.99; //actor->gamma?????????????????
                rlu::fpga::alpha = 0.1; //tau????????????????
                rlu::fpga::wsync = 1; //wsync mechanism??????????????????????
                krnl_top.setArg(0, rlu::fpga::in1_buf);
                krnl_top.setArg(1, rlu::fpga::in2_buf);
                krnl_top.setArg(2, rlu::fpga::in3_buf);
                krnl_top.setArg(3, rlu::fpga::in4_buf);
                krnl_top.setArg(4, rlu::fpga::gamma);
                krnl_top.setArg(5, rlu::fpga::alpha);
                krnl_top.setArg(6, rlu::fpga::in5_buf);
                krnl_top.setArg(7, rlu::fpga::out1_buf);
                krnl_top.setArg(8, rlu::fpga::out2_buf);
                krnl_top.setArg(9, rlu::fpga::out3_buf);
                krnl_top.setArg(10, rlu::fpga::out4_buf); //bias2, float*L3
                krnl_top.setArg(11, rlu::fpga::wsync);
                krnl_top.setArg(12, rlu::fpga::out5_buf); //Logging Qs  float*BATCHS*BSIZE
                krnl_top.setArg(13, rlu::fpga::out6_buf); //Logging Loss  float*BATCHS*BSIZE

                rlu::fpga::insert_signal_in = 0;
                rlu::fpga::update_signal = 1;
                rlu::fpga::sample_signal = 0;
                rlu::fpga::load_seed=0;
                krnl_tree.setArg(0, rlu::fpga::insert_signal_in);
                krnl_tree.setArg(1, rlu::fpga::insind_buf);
                krnl_tree.setArg(2, rlu::fpga::inpn_buf);
                krnl_tree.setArg(3, rlu::fpga::update_signal);
                krnl_tree.setArg(5, rlu::fpga::sample_signal);
                krnl_tree.setArg(6, rlu::fpga::load_seed);
                krnl_tree.setArg(7, rlu::fpga::out_buf);
                
                rlu::fpga::q.enqueueMigrateMemObjects({rlu::fpga::in1_buf,rlu::fpga::in2_buf,rlu::fpga::in3_buf,rlu::fpga::in4_buf,rlu::fpga::in5_buf}, 0);
                rlu::fpga::q.enqueueMigrateMemObjects({rlu::fpga::insind_buf,rlu::fpga::inpn_buf}, 0 /* 0 means from host*/);
                // spdlog::debug("Learner data Transferred to device");
                // rlu::fpga::q.finish();
                // printf("sent data\n");
                rlu::fpga::q.enqueueTask(krnl_top);
                rlu::fpga::q.enqueueTask(krnl_tree);
                // spdlog::debug("Learner enqued");
                // rlu::fpga::q.finish();
                // printf("enqueue\n");
                rlu::fpga::q.enqueueMigrateMemObjects({rlu::fpga::out1_buf,rlu::fpga::out2_buf,rlu::fpga::out3_buf,rlu::fpga::out4_buf,rlu::fpga::out5_buf,rlu::fpga::out6_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
                // printf("executed learner kernel with weight init\n");
                // q.enqueueMigrateMemObjects({out_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
                rlu::fpga::q.finish(); //(??? Sequential after Insert or need barrier ???):
                spdlog::debug("Learner fn q.finish");

                // watcher.lap();
                // std::cout << "Elapsed time is " << watcher.seconds() << std::endl;

                // increase the number of gradient steps ?????????????????????=====================

                // copy the weights from FPGA to the CPU
                synchronize_weights();
                this->get_update_steps(true);
                this->wake_up_actor();
            }
            // }
        }

    private:
        // initialize the bitstream
        void initialize_bitstream(const std::string &filepath) {
            // TODO: initialize the bitstream and push to the FPGA
            // cl_int err;
            std::string binaryFile = "./top.xclbin";
            unsigned fileBufSize;
            std::vector<cl::Device> devices = xcl::get_xilinx_devices();
            //devices.resize(1);
            cl::Device device = devices[0];
            std::string device_name = device.getInfo<CL_DEVICE_NAME>(&rlu::fpga::err);
            devices.resize(1);
            cl::Context context(device, NULL, NULL, NULL, &rlu::fpga::err);
            char *fileBuf = xcl::read_binary_file(binaryFile, fileBufSize);
            cl::Program::Binaries bins{{fileBuf, fileBufSize}};
            // using global variables (injecting global program and queue once)
            rlu::fpga::program = cl::Program(context, devices, bins, NULL, &rlu::fpga::err);
            rlu::fpga::q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE| CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &rlu::fpga::err);

            // IO size learners
            rlu::fpga::In_rows.resize(L1*BATCHS);
            rlu::fpga::In_rows_snt.resize(L1*BATCHS);
            rlu::fpga::Out_w1bram.resize(L1);
            rlu::fpga::Out_w2bram.resize(L2);
            rlu::fpga::In_actions.resize(BATCHS);
            rlu::fpga::In_rewards.resize(BATCHS);
            rlu::fpga::In_dones.resize(BATCHS);
            rlu::fpga::Out_bias1.resize(L2);
            rlu::fpga::Out_bias2.resize(L3);
            rlu::fpga::Out_Q.resize(BATCHS*BSIZE);
            rlu::fpga::Out_Loss.resize(BATCHS*BSIZE);
            // IO size replay
            rlu::fpga::insert_ind.resize(insert_batch);
            rlu::fpga::init_priority.resize(insert_batch);
            rlu::fpga::ind_o_out.resize(N_learner);

            // Link host buffer to DDR pointers
            rlu::fpga::InrExt.obj = rlu::fpga::In_rows.data();
            rlu::fpga::InrExt.param = 0;
            rlu::fpga::InrExt.banks = XCL_MEM_DDR_BANK0;
            rlu::fpga::InrExt.flags = XCL_MEM_DDR_BANK0;

            rlu::fpga::InrExt2.obj = rlu::fpga::In_rows_snt.data();
            rlu::fpga::InrExt2.param = 0;
            rlu::fpga::InrExt2.banks = XCL_MEM_DDR_BANK0;
            rlu::fpga::InrExt2.flags = XCL_MEM_DDR_BANK0;

            rlu::fpga::InrExt3.obj = rlu::fpga::In_actions.data();
            rlu::fpga::InrExt3.param = 0;
            rlu::fpga::InrExt3.banks = XCL_MEM_DDR_BANK0;
            rlu::fpga::InrExt3.flags = XCL_MEM_DDR_BANK0;

            rlu::fpga::InrExt4.obj = rlu::fpga::In_rewards.data();
            rlu::fpga::InrExt4.param = 0;
            rlu::fpga::InrExt4.banks = XCL_MEM_DDR_BANK0;
            rlu::fpga::InrExt4.flags = XCL_MEM_DDR_BANK0;

            rlu::fpga::InrExt5.obj = rlu::fpga::In_dones.data();
            rlu::fpga::InrExt5.param = 0;
            rlu::fpga::InrExt5.banks = XCL_MEM_DDR_BANK0;
            rlu::fpga::InrExt5.flags = XCL_MEM_DDR_BANK0;

            rlu::fpga::OutExt.obj = rlu::fpga::Out_w1bram.data();
            rlu::fpga::OutExt.param = 0;
            rlu::fpga::OutExt.banks = XCL_MEM_DDR_BANK0;
            rlu::fpga::OutExt.flags = XCL_MEM_DDR_BANK0;

            rlu::fpga::OutExt2.obj = rlu::fpga::Out_w2bram.data();
            rlu::fpga::OutExt2.param = 0;
            rlu::fpga::OutExt2.banks = XCL_MEM_DDR_BANK0;
            rlu::fpga::OutExt2.flags = XCL_MEM_DDR_BANK0;

            rlu::fpga::OutExt3.obj = rlu::fpga::Out_bias1.data();
            rlu::fpga::OutExt3.param = 0;
            rlu::fpga::OutExt3.banks = XCL_MEM_DDR_BANK0;
            rlu::fpga::OutExt3.flags = XCL_MEM_DDR_BANK0;

            rlu::fpga::OutExt4.obj = rlu::fpga::Out_bias2.data();
            rlu::fpga::OutExt4.param = 0;
            rlu::fpga::OutExt4.banks = XCL_MEM_DDR_BANK0;
            rlu::fpga::OutExt4.flags = XCL_MEM_DDR_BANK0;

            rlu::fpga::OutExt5.obj =rlu::fpga::Out_Q.data();
            rlu::fpga::OutExt5.param = 0;
            rlu::fpga::OutExt5.banks = XCL_MEM_DDR_BANK0;
            rlu::fpga::OutExt5.flags = XCL_MEM_DDR_BANK0;

            rlu::fpga::OutExt6.obj = rlu::fpga::Out_Loss.data();
            rlu::fpga::OutExt6.param = 0;
            rlu::fpga::OutExt6.banks = XCL_MEM_DDR_BANK0;
            rlu::fpga::OutExt6.flags = XCL_MEM_DDR_BANK0;

            rlu::fpga::RepInExt1.obj = rlu::fpga::insert_ind.data();
            rlu::fpga::RepInExt1.param = 0;
            rlu::fpga::RepInExt1.flags = 1|XCL_MEM_TOPOLOGY;

            rlu::fpga::RepInExt2.obj = rlu::fpga::init_priority.data();
            rlu::fpga::RepInExt2.param = 0;
            rlu::fpga::RepInExt2.flags = 1|XCL_MEM_TOPOLOGY;

            rlu::fpga::RepoutExt.obj = rlu::fpga::ind_o_out.data();
            rlu::fpga::RepoutExt.param = 0;
            rlu::fpga::RepoutExt.flags = 1|XCL_MEM_TOPOLOGY;


            // Create the buffers and allocate memory
            rlu::fpga::in1_buf = cl::Buffer(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(blockvec) * L1 * BATCHS, &rlu::fpga::InrExt, &rlu::fpga::err);
            rlu::fpga::in2_buf = cl::Buffer(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(blockvec) * L1 * BATCHS, &rlu::fpga::InrExt2, &rlu::fpga::err);
            rlu::fpga::in3_buf = cl::Buffer(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(actvec) * BATCHS, &rlu::fpga::InrExt3, &rlu::fpga::err);
            // std::cout << sizeof(actvec) * BATCHS << std::endl;
            rlu::fpga::in4_buf = cl::Buffer(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(blockvec) * BATCHS, &rlu::fpga::InrExt4, &rlu::fpga::err);
            rlu::fpga::in5_buf = cl::Buffer(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(bsbit) * BATCHS, &rlu::fpga::InrExt5, &rlu::fpga::err);
            rlu::fpga::out1_buf = cl::Buffer(context, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(w1blockvec) * L1, &rlu::fpga::OutExt, &rlu::fpga::err);
            rlu::fpga::out2_buf = cl::Buffer(context, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(w3blockvec) * L2, &rlu::fpga::OutExt2, &rlu::fpga::err);
            rlu::fpga::out3_buf = cl::Buffer(context, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(float) * L2, &rlu::fpga::OutExt3, &rlu::fpga::err);
            rlu::fpga::out4_buf = cl::Buffer(context, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(float) * L3, &rlu::fpga::OutExt4, &rlu::fpga::err);
            rlu::fpga::out5_buf = cl::Buffer(context, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(float) * BATCHS*BSIZE, &rlu::fpga::OutExt5, &rlu::fpga::err);
            rlu::fpga::out6_buf = cl::Buffer(context, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(float) * BATCHS*BSIZE, &rlu::fpga::OutExt6, &rlu::fpga::err);
            // printf("Learners data transfer buffers allocated\n");
            spdlog::debug("Learners data transfer buffers allocateds");

            rlu::fpga::insind_buf = cl::Buffer(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(int) * insert_batch, &rlu::fpga::RepInExt1, &rlu::fpga::err);
            rlu::fpga::inpn_buf = cl::Buffer(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(float) * insert_batch, &rlu::fpga::RepInExt2, &rlu::fpga::err);
            rlu::fpga::out_buf = cl::Buffer(context, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(int) * N_learner, &rlu::fpga::RepoutExt, &rlu::fpga::err);
            // printf("Replay data transfer buffers allocated\n");
            spdlog::debug("Replay data transfer buffers allocated");
            int i, j, jj;
    
            // =========================================================================================================
            // ======================================Replay Tree Initialze===============================================
            // =========================================================================================================
            cl::Kernel krnl_tree(rlu::fpga::program, "Top_tree:{Top_tree_1}");
            rlu::fpga::insert_signal_in = 1;
            rlu::fpga::update_signal=0;
            rlu::fpga::sample_signal = 0;
            rlu::fpga::load_seed = 0;
            krnl_tree.setArg(0, rlu::fpga::insert_signal_in);
            krnl_tree.setArg(1, rlu::fpga::insind_buf);
            krnl_tree.setArg(2, rlu::fpga::inpn_buf);
            krnl_tree.setArg(3, rlu::fpga::update_signal);
            krnl_tree.setArg(5, rlu::fpga::sample_signal);
            krnl_tree.setArg(6, rlu::fpga::load_seed);
            krnl_tree.setArg(7, rlu::fpga::out_buf);
            // q.enqueueMigrateMemObjects({insind_buf}, 0);
            // q.enqueueMigrateMemObjects({inpn_buf}, 0);
            rlu::fpga::q.enqueueTask(krnl_tree);
            // q.enqueueMigrateMemObjects({out_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
            rlu::fpga::q.finish();
            // printf("\nFrom Host: Tree init done.\n");
            spdlog::debug("From Host: Tree init done.");

            // =========================================================================================================
            // =======================================Learners weights Initialze==========================================
            // =========================================================================================================
    
            for (jj = 0; jj < BATCHS; jj++) {   
                for (j = 0; j < L1; j++) {
                    for (i = 0; i < BSIZE; i++) {
                    rlu::fpga::In_rows[L1*jj+j].a[i] = 0;
                    rlu::fpga::In_rows_snt[L1*jj+j].a[i] = 0;
                    // printf("%f ",rlu::fpga::In_rows[j].a[i]);
                    // printf("%f ",In_rows_snt[j].a[i]);
                    }
                }
            }

            for (jj = 0; jj < BATCHS; jj++) {   
                for (i = 0; i < BSIZE; i++) {
                // printf("\njj,i:%d,%d\n",jj,i);
                    rlu::fpga::In_actions[jj].a[i] = 0;
                    rlu::fpga::In_rewards[jj].a[i] = 0;
                    rlu::fpga::In_dones[jj].a[i] = 0;
                    // printf("%d ",rlu::fpga::In_actions[jj].a[i]);
                }
            }

            spdlog::debug("Input inied for w init.");
            cl::Kernel krnl_top0(rlu::fpga::program, "learners_top:{top_1}");
            cl::Kernel krnl_tree0(rlu::fpga::program, "Top_tree"); 
            rlu::fpga::gamma=0;
            rlu::fpga::alpha=0;
            rlu::fpga::wsync = 0;
            krnl_top0.setArg(0, rlu::fpga::in1_buf);
            krnl_top0.setArg(1, rlu::fpga::in2_buf);
            krnl_top0.setArg(2, rlu::fpga::in3_buf);
            krnl_top0.setArg(3, rlu::fpga::in4_buf);
            krnl_top0.setArg(4, rlu::fpga::gamma);
            krnl_top0.setArg(5, rlu::fpga::alpha);
            krnl_top0.setArg(6, rlu::fpga::in5_buf);
            krnl_top0.setArg(7, rlu::fpga::out1_buf);
            krnl_top0.setArg(8, rlu::fpga::out2_buf);
            krnl_top0.setArg(9, rlu::fpga::out3_buf);
            krnl_top0.setArg(10, rlu::fpga::out4_buf);
            krnl_top0.setArg(11, rlu::fpga::wsync);
            krnl_top0.setArg(12, rlu::fpga::out5_buf); 
            krnl_top0.setArg(13, rlu::fpga::out6_buf); 

            rlu::fpga::insert_signal_in = 0;
            rlu::fpga::update_signal=1;
            rlu::fpga::sample_signal = 0;
            rlu::fpga::load_seed=0;
            krnl_tree0.setArg(0, rlu::fpga::insert_signal_in);
            krnl_tree0.setArg(1, rlu::fpga::insind_buf);
            krnl_tree0.setArg(2, rlu::fpga::inpn_buf);
            krnl_tree0.setArg(3, rlu::fpga::update_signal);
            krnl_tree0.setArg(5, rlu::fpga::sample_signal);
            krnl_tree0.setArg(6, rlu::fpga::load_seed);
            krnl_tree0.setArg(7, rlu::fpga::out_buf);
            
            rlu::fpga::q.enqueueMigrateMemObjects({rlu::fpga::in1_buf,rlu::fpga::in2_buf,rlu::fpga::in3_buf,rlu::fpga::in4_buf,rlu::fpga::in5_buf}, 0 /* 0 means from host*/);
            // rlu::fpga::q.enqueueMigrateMemObjects({insind_buf,inpn_buf}, 0 /* 0 means from host*/);
            spdlog::debug("input transferred for w init.");
            rlu::fpga::q.finish();
            // printf("sent data\n");
            rlu::fpga::q.enqueueTask(krnl_top0);
            rlu::fpga::q.enqueueTask(krnl_tree0);
            spdlog::debug("kernels executed for w init.");
            rlu::fpga::q.finish();

            // q.enqueueMigrateMemObjects({out1_buf,out2_buf,out3_buf,out4_buf,out5_buf,out6_buf}, CL_MIGRATE_MEM_OBJECT_HOST);

            rlu::fpga::q.finish(); 
            spdlog::debug("initialize_bitstream done.");
            // =========================================================================================================
            // ==================================Checking weight tensors shapes==========================================
            // =========================================================================================================
            // fmt::print("Number of layers {}", actor->parameters().size());
            // for (auto &param: actor->parameters()) {
            //     fmt::print("{}\n", param);
            // }
            // fmt::print("OBS dimension", actor->train_step.obs.size()); 
            // fmt::print("NEXT_OBS dimension", actor->train_step.next_obs.size()); 
            // throw std::runtime_error("Stop here");
        }



        void synchronize_weights() {
            // synchronize the weights from the FPGA to the CPU. The CPU weights is in
            // actor->parameters(). Actor is a torch::nn::Module. It is a 3-layer MLP with relu activation
            // The weights can be referred by index. For example, actor->parameters()[0], etc

            // W1
            for(int i = 0; i < L1; i++) {
                for(int j = 0; j < L2; j++) {
                    // printf("%.8f ",rlu::fpga::Out_w1bram[i].a[j]);  //L1 rows, L2 cols     
                    actor->parameters()[0].index_put_({j,i},rlu::fpga::Out_w1bram[i].a[j]);         
                }
                printf("\n");        
            }
            // W2
            for(int i = 0; i < L2; i++) {
                for(int j = 0; j < L3; j++) {
                    // printf("%.8f ",rlu::fpga::Out_w2bram[i].a[j]);  //L1 rows, L2 cols     
                    // actor->parameters()[2][j][i]=rlu::fpga::Out_w1bram[i].a[j];     
                    actor->parameters()[2].index_put_({j,i},rlu::fpga::Out_w2bram[i].a[j]);    
                }
                printf("\n");        
            }
            // bias1
            for(int i = 0; i < L2; i++) {
                // printf("%.8f ",rlu::fpga::Out_bias1[i]);  //L1 rows, L2 cols     
                // actor->parameters()[1][i]=rlu::fpga::Out_bias1[i];  
                actor->parameters()[1].index_put_({i},rlu::fpga::Out_bias1[i]); 
                printf("\n");        
            }
            // bias2
            for(int i = 0; i < L3; i++) {
                // printf("%.8f ",rlu::fpga::Out_bias2[i]);  //L1 rows, L2 cols     
                // actor->parameters()[3][i]=rlu::fpga::Out_bias2[i];  
                actor->parameters()[3].index_put_({i},rlu::fpga::Out_bias2[i]);        
                printf("\n");        
            }
            spdlog::debug("sync weights done.");
        }

    };

}


#endif //HIPC21_OFF_POLICY_TRAINER_FPGA_H
