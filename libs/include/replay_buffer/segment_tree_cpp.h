//
// Created by Chi Zhang on 9/21/21.
//

#ifndef HIPC21_SEGMENT_TREE_CPP_H
#define HIPC21_SEGMENT_TREE_CPP_H

#include "segment_tree_base.h"

namespace rlu::replay_buffer {
    class SegmentTreeCPP : public SegmentTree {
    public:
        SegmentTreeCPP() = default;

        virtual ~SegmentTreeCPP();

        explicit SegmentTreeCPP(int64_t size);

        [[nodiscard]] int64_t size() const override;

        torch::Tensor operator[](const torch::Tensor &idx) const override;

        void set(const torch::Tensor &idx, const torch::Tensor &value) override;

        [[nodiscard]] float reduce(int64_t start, int64_t end) const override;

        [[nodiscard]] float reduce(int64_t end) const;

        [[nodiscard]] float reduce() const override;

        [[nodiscard]] torch::Tensor get_prefix_sum_idx(torch::Tensor value) const override;

    protected:
        int64_t m_size{};
        int64_t m_bound{};
        float *m_values{};

        void initialize();

        // helper functions

        [[nodiscard]] virtual int64_t convert_to_node_idx(int64_t data_idx) const;

        [[nodiscard]] virtual int64_t convert_to_data_idx(int64_t node_idx) const;

        [[nodiscard]] virtual int64_t get_parent(int64_t node_idx) const;

        [[nodiscard]] virtual int64_t get_sibling(int64_t node_idx) const;

        [[nodiscard]] virtual bool is_leaf(int64_t node_idx) const;

        [[nodiscard]] virtual bool is_left(int64_t node_idx) const;

        [[nodiscard]] virtual bool is_right(int64_t node_idx) const;

        [[nodiscard]] virtual int64_t get_left_child(int64_t node_idx) const;

        [[nodiscard]] virtual int64_t get_right_child(int64_t node_idx) const;

        [[nodiscard]] virtual int64_t get_root() const;

        [[nodiscard]] virtual float get_value(int64_t node_idx) const;

        virtual void set_value(int64_t node_idx, float value);
    };
}

#endif //HIPC21_SEGMENT_TREE_CPP_H
