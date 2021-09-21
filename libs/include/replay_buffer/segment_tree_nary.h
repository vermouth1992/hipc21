//
// Created by Chi Zhang on 9/21/21.
//

#ifndef HIPC21_SEGMENT_TREE_NARY_H
#define HIPC21_SEGMENT_TREE_NARY_H

#include "segment_tree_base.h"

namespace rlu::replay_buffer {
    class SegmentTreeNary : public SegmentTree {
    public:
        explicit SegmentTreeNary(int64_t size, int64_t n = 8);

        virtual ~SegmentTreeNary() { delete[]m_values; }

        torch::Tensor operator[](const torch::Tensor &idx) const override;

        void set(const torch::Tensor &idx, const torch::Tensor &value) override;

        [[nodiscard]] float reduce() const override;

        [[nodiscard]] float reduce(int64_t start, int64_t end) const override;

        [[nodiscard]] float reduce(int64_t end) const;

        [[nodiscard]] torch::Tensor get_prefix_sum_idx(torch::Tensor value) const override;

    protected:
        [[nodiscard]] int64_t size() const override;

        [[nodiscard]] virtual inline int64_t get_node_idx_after_padding(int64_t node_idx) const;

        [[nodiscard]] virtual inline float get_value(int64_t node_idx) const;

        virtual inline void set_value(int64_t node_idx, float value);

        [[nodiscard]] virtual inline int64_t convert_to_node_idx(int64_t data_idx) const;

        [[nodiscard]] virtual inline int64_t convert_to_data_idx(int64_t node_idx) const;

        [[nodiscard]] virtual inline int64_t get_parent(int64_t node_idx) const;

        [[nodiscard]] virtual inline int64_t get_root() const;

        [[nodiscard]] virtual inline bool is_leaf(int64_t node_idx) const;

        // get the most left child in a N-ary heap in zero-based array
        [[nodiscard]] virtual inline int64_t get_left_child(int64_t node_idx) const;

        void initialize();

    private:
        int64_t m_n{};
        int64_t log2_m_n{};
        int64_t m_size{};
        // the size of the last level
        int64_t last_level_size{};
        int64_t m_bound{};
        int64_t m_padding{};
        float *m_values{};

    };
}


#endif //HIPC21_SEGMENT_TREE_NARY_H
