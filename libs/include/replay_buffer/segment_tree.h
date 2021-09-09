//
// Created by chi on 7/3/21.
//

#ifndef HIPC21_SEGMENT_TREE_H
#define HIPC21_SEGMENT_TREE_H

#include <torch/torch.h>
#include <vector>
#include "nn/functional.h"
#include "exception.h"

namespace rlu::replay_buffer {
    class SegmentTree {
    public:
        [[nodiscard]] virtual auto size() const -> int64_t = 0;

        virtual auto operator[](const torch::Tensor &idx) const -> torch::Tensor = 0;

        virtual auto set(const torch::Tensor &idx, const torch::Tensor &value) -> void = 0;

        [[nodiscard]] virtual auto reduce(int64_t start, int64_t end) const -> float = 0;

        [[nodiscard]] virtual auto reduce() const -> float = 0;

        [[nodiscard]] virtual auto get_prefix_sum_idx(torch::Tensor value) const -> torch::Tensor = 0;

        // for compatible with the abstraction of the FPGA segment tree
        [[nodiscard]] virtual auto sample_idx(int64_t batch_size) const -> torch::Tensor;
    };

    class SegmentTreeFPGA final : public SegmentTree {
    public:
        explicit SegmentTreeFPGA(int64_t size) {

        }

        // query the FPGA about the size of the segment tree.
        [[nodiscard]] int64_t size() const override {
            throw NotImplemented();
        }

        // query the FPGA about the weights in the segment tree
        torch::Tensor operator[](const torch::Tensor &idx) const override {
            throw NotImplemented();
        }

        void set(const torch::Tensor &idx, const torch::Tensor &value) override {
            // TODO: set the priority of idx to value on the FPGA. This function must be implemented
            // tensor shape: one dimension array idx: M, value M
            throw NotImplemented();
        }

        [[nodiscard]] float reduce(int64_t start, int64_t end) const override {
            throw NotImplemented();
        }

        [[nodiscard]] float reduce() const override {
            throw NotImplemented();
        }

        [[nodiscard]] torch::Tensor get_prefix_sum_idx(torch::Tensor value) const override {
            throw NotImplemented();
        }

        [[nodiscard]] torch::Tensor sample_idx(int64_t batch_size) const override {
            // TODO: this function must be implemented
            throw NotImplemented();
        }

    };

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

        [[nodiscard]] virtual inline int64_t convert_to_node_idx(int64_t data_idx) const;

        [[nodiscard]] virtual inline int64_t convert_to_data_idx(int64_t node_idx) const;

        [[nodiscard]] virtual inline int64_t get_parent(int64_t node_idx) const;

        [[nodiscard]] virtual inline int64_t get_sibling(int64_t node_idx) const;

        [[nodiscard]] virtual inline bool is_leaf(int64_t node_idx) const;

        [[nodiscard]] virtual inline bool is_left(int64_t node_idx) const;

        [[nodiscard]] virtual inline bool is_right(int64_t node_idx) const;

        [[nodiscard]] virtual inline int64_t get_left_child(int64_t node_idx) const;

        [[nodiscard]] virtual inline int64_t get_right_child(int64_t node_idx) const;

        [[nodiscard]] virtual inline int64_t get_root() const;

        [[nodiscard]] virtual inline float get_value(int64_t node_idx) const;

        virtual inline void set_value(int64_t node_idx, float value);
    };


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


    class SegmentTreeCPPOpt : public SegmentTreeCPP {
    public:
        explicit SegmentTreeCPPOpt(int64_t size, int64_t partition_height);

    protected:
        [[nodiscard]] inline int64_t convert_to_node_idx(int64_t data_idx) const override;

        [[nodiscard]] inline int64_t convert_to_data_idx(int64_t node_idx) const override;

        [[nodiscard]] inline int64_t get_parent(int64_t node_idx) const override;

        [[nodiscard]] inline int64_t get_left_child(int64_t node_idx) const override;

        [[nodiscard]] inline int64_t get_right_child(int64_t node_idx) const override;

    private:
        int64_t m_partition_height;
        int64_t m_block_height;
        int64_t m_block_branch_factor;
        int64_t m_bottom_left_block_idx;

        // given a node index, return the block index (0-based)
        [[nodiscard]] inline int64_t get_block_index(int64_t node_idx) const;

        // given a block index, return the parent block index
        [[nodiscard]] inline int64_t get_parent_block_index(int64_t block_idx) const;

        [[nodiscard]] inline int64_t get_second_row_first_element_inside_block(int64_t block_idx) const;

        [[nodiscard]] inline int64_t get_last_row_first_element_inside_block(int64_t block_idx) const;

        [[nodiscard]] inline int64_t get_parent_block_idx(int64_t block_idx) const;
    };

}
#endif //HIPC21_SEGMENT_TREE_H
