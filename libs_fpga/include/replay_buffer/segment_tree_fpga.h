//
// Created by Chi Zhang on 9/21/21.
//

#ifndef HIPC21_SEGMENT_TREE_FPGA_H
#define HIPC21_SEGMENT_TREE_FPGA_H

#include "replay_buffer/segment_tree_base.h"

namespace rlu::replay_buffer {
    class SegmentTreeFPGA final : public SegmentTree {
    public:
        explicit SegmentTreeFPGA(int64_t size) {

        }

        // query the FPGA about the size of the segment tree.
        [[nodiscard]] int64_t size() const override {
            throw NotImplemented();
        }

        // query the FPGA about the weights in the segment tree
        torch::Tensor operator[](__attribute__ ((unused)) const torch::Tensor &idx) const override {
            throw NotImplemented();
        }

        void set(const torch::Tensor &idx, const torch::Tensor &value) override {
            // TODO: set the priority of idx to value on the FPGA. This function must be implemented
            // tensor shape: one dimension array idx: M, value M
//            throw NotImplemented();
        }

        [[nodiscard]] float reduce(__attribute__ ((unused)) int64_t start,
                                   __attribute__ ((unused)) int64_t end) const override {
            throw NotImplemented();
        }

        [[nodiscard]] float reduce() const override {
            throw NotImplemented();
        }

        [[nodiscard]] torch::Tensor get_prefix_sum_idx(__attribute__ ((unused)) torch::Tensor value) const override {
            throw NotImplemented();
        }

        [[nodiscard]] torch::Tensor sample_idx(int64_t batch_size) const override {
            // TODO: this function must be implemented
            return torch::zeros({batch_size}, torch::TensorOptions().dtype(torch::kInt64));
        }

    };
}


#endif //HIPC21_SEGMENT_TREE_FPGA_H
