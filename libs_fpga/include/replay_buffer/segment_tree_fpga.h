//
// Created by Chi Zhang on 9/21/21.
//

#ifndef HIPC21_SEGMENT_TREE_FPGA_H
#define HIPC21_SEGMENT_TREE_FPGA_H

#include "replay_buffer/segment_tree_base.h"
#include "fpga.h"

#include "rmm.h"
#include "topl_new.h"



namespace rlu::replay_buffer {
    class SegmentTreeFPGA final : public SegmentTree {
    public:
        explicit SegmentTreeFPGA(int64_t size) {

        }

        // query the FPGA about the size of the segment tree.
        [[nodiscard]] int64_t size() const override {
            throw NotImplemented("SegmentTreeFPGA::size");
        }

        // query the FPGA about the weights in the segment tree
        torch::Tensor operator[](__attribute__ ((unused)) const torch::Tensor &idx) const override {
            return torch::zeros_like(idx);
        }

        void set(const torch::Tensor &idx, const torch::Tensor &value) override {
            // TODO: set the priority of idx to value on the FPGA. This function must be implemented
            // tensor shape: one dimension array idx: M, value M
//            throw NotImplemented();
            for (int i = 0; i < insert_batch; i++) {
                // printf("\njj,i:%d,%d\n",jj,i);
                rlu::fpga::insert_ind[i] = idx.index({i}).item<float>(); 
                rlu::fpga::init_priority[i] = value.index({i}).item<float>();
                // printf("%f ",In_actions[jj].a[i]);
            }

        }

        [[nodiscard]] float reduce(__attribute__ ((unused)) int64_t start,
                                   __attribute__ ((unused)) int64_t end) const override {
            throw NotImplemented("SegmentTreeFPGA::reduce");
        }

        [[nodiscard]] float reduce() const override {
            return 1.;
            // throw NotImplemented("SegmentTreeFPGA::reduce");
        }

        [[nodiscard]] torch::Tensor get_prefix_sum_idx(__attribute__ ((unused)) torch::Tensor value) const override {
            throw NotImplemented("SegmentTreeFPGA::get_prefix_sum_idx");
        }
 
        [[nodiscard]] torch::Tensor sample_idx(int64_t batch_size) const override { //Int32???????
            // TODO: this function must be implemented
            cl::Kernel krnl_tree2(rlu::fpga::program, "Top_tree");
            // Doing inser and sampling together in one kernel call
            rlu::fpga::insert_signal_in = 2;
            rlu::fpga::update_signal=0;
            rlu::fpga::sample_signal = 1;
            rlu::fpga::load_seed = 1;

            krnl_tree2.setArg(0, rlu::fpga::insert_signal_in);
            krnl_tree2.setArg(1, rlu::fpga::insind_buf);
            krnl_tree2.setArg(2, rlu::fpga::inpn_buf);
            krnl_tree2.setArg(3, rlu::fpga::update_signal);
            krnl_tree2.setArg(5, rlu::fpga::sample_signal);
            krnl_tree2.setArg(6, rlu::fpga::load_seed);
            krnl_tree2.setArg(7, rlu::fpga::out_buf);
            rlu::fpga::q.enqueueMigrateMemObjects({rlu::fpga::insind_buf,rlu::fpga::inpn_buf}, 0);
            rlu::fpga::q.finish();
            rlu::fpga::q.enqueueTask(krnl_tree2);
            rlu::fpga::q.finish();
            rlu::fpga::q.enqueueMigrateMemObjects({rlu::fpga::out_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
            rlu::fpga::q.finish();

            torch::Tensor ret_tensor = torch::zeros({batch_size}, torch::TensorOptions().dtype(torch::kInt64));
            for(int i = 0; i < batch_size; i++) { //confirm batch size = N_learners
                // sample_idx[i]=rlu::fpga::ind_o_out[i];
                ret_tensor.index_put_({i},rlu::fpga::ind_o_out[i]); 
            } 
            
            // return torch::zeros({batch_size}, torch::TensorOptions().dtype(torch::kInt64));
            return ret_tensor;
        }

    };
}


#endif //HIPC21_SEGMENT_TREE_FPGA_H
