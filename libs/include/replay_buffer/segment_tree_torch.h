//
// Created by Chi Zhang on 9/21/21.
//

#ifndef HIPC21_SEGMENT_TREE_TORCH_H
#define HIPC21_SEGMENT_TREE_TORCH_H

#include "segment_tree_base.h"

namespace rlu::replay_buffer {

    // directly pytorch implementation using Tensor batch operation
    class SegmentTreeTorch final : public SegmentTree {
    public:
        explicit SegmentTreeTorch(int64_t size);

        [[nodiscard]] int64_t size() const override;

        torch::Tensor operator[](const torch::Tensor &idx) const override;

        void set(const torch::Tensor &idx, const torch::Tensor &value) override;

        [[nodiscard]] float reduce(int64_t start, int64_t end) const override;

        [[nodiscard]] float reduce() const override;

        [[nodiscard]] torch::Tensor get_prefix_sum_idx(torch::Tensor value) const override;


    private:
        int64_t m_size;
        int64_t m_bound;
        torch::Tensor m_values;
    };

}
#endif //HIPC21_SEGMENT_TREE_TORCH_H
