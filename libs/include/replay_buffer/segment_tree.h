//
// Created by chi on 7/3/21.
//

#ifndef HIPC21_SEGMENT_TREE_H
#define HIPC21_SEGMENT_TREE_H

#include <torch/torch.h>
#include <vector>
#include "utils/rl_functional.h"

namespace rlu::replay_buffer {

    class SegmentTreeTorch {
    public:
        explicit SegmentTreeTorch(int64_t size) : m_size(size) {
            m_bound = 1;
            while (m_bound < size) {
                m_bound = m_bound * 2;
            }
            m_values = torch::zeros({m_bound * 2}, torch::TensorOptions().dtype(torch::kFloat32));
        }

        int64_t size() const {
            return m_size;
        }

        std::shared_ptr<torch::Tensor> operator[](const torch::Tensor &idx) const {
            return std::make_unique<torch::Tensor>(m_values.index({idx + m_bound}));
        }

        void set(const torch::Tensor &idx, const torch::Tensor &value) {
            torch::Tensor idx_bound = idx + m_bound;
            m_values.index_put_({idx_bound}, value);
            while (idx_bound.index({0}).item().toInt() > 1) {
                idx_bound = torch::div(idx_bound, 2, "floor");
                auto left = m_values.index({idx_bound * 2});
                auto right = m_values.index({idx_bound * 2 + 1});
                m_values.index_put_({idx_bound}, left + right);
            }
        }

        float reduce(int64_t start, int64_t end) const {
            start = start + m_bound - 1;
            end = end + m_bound;
            float result = 0.;
            while (end - start > 1) {
                if (start % 2 == 0) {
                    result += m_values.index({start + 1}).item().toFloat();
                }
                start = start / 2;
                if (end % 2 == 1) {
                    result += m_values.index({end - 1}).item().toFloat();
                }
                end = end / 2;
            }
            return result;
        }

        float reduce() const {
            return m_values.index({1}).item().toFloat();
        }

        std::shared_ptr<torch::Tensor> get_prefix_sum_idx(torch::Tensor value) const {
            auto index = std::make_unique<torch::Tensor>(
                    torch::ones_like(value, torch::TensorOptions().dtype(torch::kInt64)));
            while (index->index({0}).item().toInt() < m_bound) {
                *index = *index * 2;
                auto lsons = m_values.index({*index});
                auto direct = torch::less(lsons, value);
                value = value - lsons * direct;
                *index = *index + direct;
            }
            *index = *index - m_bound;
            return index;
        }


    private:
        int64_t m_size;
        int64_t m_bound;
        torch::Tensor m_values;
    };

    // convert a 1D pytorch array to std::vector
    template<typename T>
    std::shared_ptr<std::vector<T>> convert_tensor_to_vector(const torch::Tensor &tensor) {
        return std::make_shared<std::vector<T>>(tensor.data_ptr<T>(), tensor.data_ptr<T>() + tensor.numel());
    }


    class SegmentTreeCPP {
    public:
        SegmentTreeCPP() = default;

        ~SegmentTreeCPP() {
            delete[]m_values;
        }

        explicit SegmentTreeCPP(int64_t size) : m_size(size) {
            m_bound = 1;
            while (m_bound < size) {
                m_bound = m_bound * 2;
            }
            initialize();
        }

        int64_t size() const {
            return m_size;
        }

        virtual inline int64_t convert_to_node_idx(int64_t data_idx) const {
            return data_idx + m_bound;
        }

        virtual inline int64_t convert_to_data_idx(int64_t node_idx) const {
            return node_idx - m_bound;
        }

        virtual inline int64_t get_parent(int64_t node_idx) const {
            return node_idx / 2;
        }

        virtual inline int64_t get_sibling(int64_t node_idx) const {
            if (node_idx % 2 == 0) {
                return node_idx + 1;
            } else {
                return node_idx - 1;
            }
        }

        virtual inline bool is_leaf(int64_t node_idx) const {
            int64_t left_child = get_left_child(node_idx);
            return left_child >= 2 * m_bound;
        }

        virtual inline bool is_left(int64_t node_idx) const {
            return node_idx % 2 == 0;
        }

        virtual inline bool is_right(int64_t node_idx) const {
            return node_idx % 2 == 1;
        }

        virtual inline int64_t get_left_child(int64_t node_idx) const {
            return node_idx * 2;
        }

        virtual inline int64_t get_right_child(int64_t node_idx) const {
            return node_idx * 2 + 1;
        }

        virtual inline int64_t get_root() const {
            return 1;
        }

        virtual inline float get_value(int64_t node_idx) const {
            auto value = m_values[node_idx];
            return value;
        }

        virtual inline void set_value(int64_t node_idx, float value) {
            m_values[node_idx] = value;
        }

        std::shared_ptr<torch::Tensor> operator[](const torch::Tensor &idx) const {
            auto idx_vector = convert_tensor_to_vector<int64_t>(idx);
            auto output = torch::zeros_like(idx, torch::TensorOptions().dtype(torch::kFloat32));
            for (int i = 0; i < (int) idx_vector->size(); ++i) {
                output.index_put_({i}, get_value(convert_to_node_idx(idx_vector->at(i))));
            }
            return std::make_shared<torch::Tensor>(output);
        }

        void set(const torch::Tensor &idx, const torch::Tensor &value) {
            auto idx_vec = convert_tensor_to_vector<int64_t>(idx);
            auto value_vec = convert_tensor_to_vector<float>(value);
            // put all the values
            for (int i = 0; i < (int) idx_vec->size(); ++i) {
                // get data pos
                int64_t pos = idx_vec->operator[](i);
                // get node pos
                pos = convert_to_node_idx(pos);
                // set the value of the leaf node
                auto original_value = get_value(pos);
                auto new_value = value_vec->operator[](i);
                auto delta = new_value - original_value;
                // update the parent
                while (true) {
                    set_value(pos, get_value(pos) + delta);
                    if (pos == get_root()) {
                        break;
                    }
                    pos = get_parent(pos);
                }
            }
        }

        float reduce(int64_t start, int64_t end) const {
            assert(start >= 0 && end <= size() && end >= start);
            if (start == 0) {
                return reduce(end);
            } else return reduce(end) - reduce(start);
        }

        float reduce(int64_t end) const {
            assert(end > 0 && end <= size());
            if (end == size()) {
                return reduce();
            }
            end = convert_to_node_idx(end);
            float result = 0.;
            while (end != get_root()) {
                if (is_right(end)) {
                    result += get_value(get_sibling(end));
                }
                end = get_parent(end);
            }
            return result;
        }

        float reduce() const {
            return get_value(get_root());
        }

        std::shared_ptr<torch::Tensor> get_prefix_sum_idx(const torch::Tensor &value) const {
            auto value_vec = convert_tensor_to_vector<float>(value);
            auto index = std::make_unique<torch::Tensor>(
                    torch::ones_like(value, torch::TensorOptions().dtype(torch::kInt64)));

            for (int i = 0; i < (int) value_vec->size(); i++) {
                int64_t idx = get_root();
                float current_val = (*value_vec)[i];
                while (!is_leaf(idx)) {
                    idx = get_left_child(idx);
                    auto lsons = get_value(idx);
                    if (lsons < current_val) {
                        current_val -= lsons;
                        idx = get_sibling(idx);
                    }
                }
                (*index).index_put_({i}, convert_to_data_idx(idx));
            }

            return index;
        }

    protected:
        int64_t m_size{};
        int64_t m_bound{};
        float *m_values{};

        void initialize() {
            //        std::cout << m_bound << std::endl;
            m_values = new float[m_bound * 2];
            for (int i = 0; i < m_bound * 2; ++i) {
                m_values[i] = 0.;
            }
        }
    };


    class SegmentTreeNary {
    public:
        explicit SegmentTreeNary(int64_t size, int64_t n) :
                m_n(n),
                m_size(size) {
            last_level_size = 1;
            while (last_level_size < size) {
                last_level_size = last_level_size * m_n;
            }
            m_bound = (last_level_size - 1) / (m_n - 1);
            initialize();
        }

        ~SegmentTreeNary() { delete[]m_values; }

        int64_t size() const {
            return m_size;
        }

        virtual inline int64_t get_node_idx_after_padding(int64_t node_idx) const {
            return node_idx + m_padding;
        }

        virtual inline float get_value(int64_t node_idx) const {
            node_idx = get_node_idx_after_padding(node_idx);
            auto value = m_values[node_idx];
            return value;
        }

        virtual inline void set_value(int64_t node_idx, float value) {
            node_idx = get_node_idx_after_padding(node_idx);
            m_values[node_idx] = value;
        }

        virtual inline int64_t convert_to_node_idx(int64_t data_idx) const {
            return data_idx + m_bound;
        }

        virtual inline int64_t convert_to_data_idx(int64_t node_idx) const {
            return node_idx - m_bound;
        }

        virtual inline int64_t get_parent(int64_t node_idx) const {
            return (node_idx - 1) >> log2_m_n;
        }

        virtual inline int64_t get_root() const {
            return 0;
        }

        std::shared_ptr<torch::Tensor> operator[](const torch::Tensor &idx) const {
            auto idx_vector = convert_tensor_to_vector<int64_t>(idx);
            auto output = torch::zeros_like(idx, torch::TensorOptions().dtype(torch::kFloat32));
            for (int i = 0; i < (int) idx_vector->size(); ++i) {
                output.index_put_({i}, get_value(convert_to_node_idx(idx_vector->at(i))));
            }
            return std::make_shared<torch::Tensor>(output);
        }

        void set(const torch::Tensor &idx, const torch::Tensor &value) {
            auto idx_vec = convert_tensor_to_vector<int64_t>(idx);
            auto value_vec = convert_tensor_to_vector<float>(value);
            // put all the values
            for (int i = 0; i < (int) idx_vec->size(); ++i) {
                // get data pos
                int64_t pos = idx_vec->operator[](i);
                // get node pos
                pos = convert_to_node_idx(pos);
                // set the value of the leaf node
                auto original_value = get_value(pos);
                auto new_value = value_vec->operator[](i);
                auto delta = new_value - original_value;
                // update the parent
                while (true) {
                    set_value(pos, get_value(pos) + delta);
                    if (pos == get_root()) {
                        break;
                    }
                    pos = get_parent(pos);
                }
            }
        }

        float reduce() const {
            return get_value(get_root());
        }

        float reduce(int64_t start, int64_t end) const {
            assert(start >= 0 && end <= size() && end >= start);
            if (start == 0) {
                return reduce(end);
            } else return reduce(end) - reduce(start);
        }

        float reduce(int64_t end) const {
            assert(end > 0 && end <= size());
            if (end == size()) {
                return reduce();
            }
            end = convert_to_node_idx(end);
            float result = 0.;
            while (end != get_root()) {
                // sum all the node left to it.
                int64_t parent = get_parent(end);
                int64_t left_child = get_left_child(parent);
                while (true) {
                    if (left_child != end) {
                        result += get_value(left_child);
                    } else {
                        break;
                    }
                    left_child += 1;
                }
                end = parent;
            }
            return result;
        }

        virtual inline bool is_leaf(int64_t node_idx) const {
            return node_idx >= m_bound;
        }

        // get the most left child in a N-ary heap in zero-based array
        virtual inline int64_t get_left_child(int64_t node_idx) const {
            // using shift operator is crucial
            return (node_idx << log2_m_n) + 1;
        }

        std::shared_ptr<torch::Tensor> get_prefix_sum_idx(const torch::Tensor &value) const {
            auto value_vec = convert_tensor_to_vector<float>(value);
            auto index = std::make_unique<torch::Tensor>(
                    torch::ones_like(value, torch::TensorOptions().dtype(torch::kInt64)));

            for (int i = 0; i < (int) value_vec->size(); i++) {
                int64_t idx = get_root();
                float current_val = (*value_vec)[i];
                while (!is_leaf(idx)) {
                    idx = get_left_child(idx);
                    float partial_sum = 0.;
                    for (int64_t j = 0; j < m_n; ++j) {
                        float after_sum = get_value(idx) + partial_sum;
                        if (after_sum >= current_val) {
                            break;
                        }
                        // get next sibling
                        partial_sum = after_sum;
                        idx += 1;
                    }
                    current_val -= partial_sum;
                }
                (*index).index_put_({i}, convert_to_data_idx(idx));
            }

            return index;
        }

    private:
        int64_t m_n{};
        int64_t log2_m_n{};
        int64_t m_size{};
        // the size of the last level
        int64_t last_level_size{};
        int64_t m_bound{};
        int64_t m_padding{};
        float *m_values{};

        void initialize() {
            // zero-based indexing
            int64_t total_size = (last_level_size * m_n - 1) / (m_n - 1);
            // making the data at each level cache aligned
            m_padding = m_n - 1;
            log2_m_n = (int64_t) std::log2(m_n);
            m_values = new float[total_size + m_padding];
            for (int i = 0; i < total_size; ++i) {
                m_values[i] = 0.;
            }
            //        std::cout << m_n << " " << m_size << " " << last_level_size << " " << m_bound << std::endl;
        }
    };


    class SegmentTreeCPPOpt : public SegmentTreeCPP {
    public:
        explicit SegmentTreeCPPOpt(int64_t size, int64_t partition_height) :
                m_partition_height(partition_height),
                m_block_branch_factor(1 << (partition_height - 1)) {
            m_size = size;
            m_bound = 1;
            int64_t height = 1;
            while (m_bound < size || (height - 1) % (partition_height - 1) != 0) {
                m_bound = m_bound * 2;
                height += 1;
            }
            m_block_height = (height - 1) / (partition_height - 1);
            m_bottom_left_block_idx = ((1 << ((partition_height - 1) * (m_block_height - 1))) - 1)
                                      / ((1 << (partition_height - 1)) - 1);
            initialize();
        }

        inline int64_t convert_to_node_idx(int64_t data_idx) const override {
            // get the block offset towards the bottom left block
            int64_t block_offset = data_idx >> (m_partition_height - 1);
            int64_t block_idx = block_offset + m_bottom_left_block_idx;
            // get the index offset
            int64_t index_offset = data_idx % m_block_branch_factor;
            // compute the index
            int64_t block_bottom_left_node_idx = get_last_row_first_element_inside_block(block_idx);
            return block_bottom_left_node_idx + index_offset;
        }

        inline int64_t convert_to_data_idx(int64_t node_idx) const override {
            // node_idx must be leaf node
            // compute block index
            int64_t block_idx = get_block_index(node_idx);
            int64_t block_offset = block_idx - m_bottom_left_block_idx;
            int64_t block_bottom_left_node_idx = get_last_row_first_element_inside_block(block_idx);
            int64_t index_offset = node_idx - block_bottom_left_node_idx;
            // compute offset index
            return block_offset * m_block_branch_factor + index_offset;
        }

        inline int64_t get_parent(int64_t node_idx) const override {
            if (node_idx == 2 || node_idx == 3) return 1;
            // get block index
            int64_t block_idx = get_block_index(node_idx);
            int64_t block_second_row_first_node_idx = get_second_row_first_element_inside_block(block_idx);
            if (node_idx > block_second_row_first_node_idx + 1) {
                // not the second row
                int64_t offset = block_second_row_first_node_idx - 2;
                int64_t normalized_node_idx = node_idx - offset;
                return normalized_node_idx / 2 + offset;
            } else {
                // the second row. the parent is the last row of the parent block
                int64_t parent_block_idx = get_parent_block_idx(block_idx);
                int64_t next_level_start_block_idx = m_block_branch_factor * parent_block_idx + 1;
                int64_t block_bottom_left_node_idx = get_last_row_first_element_inside_block(parent_block_idx);
                return block_bottom_left_node_idx + block_idx - next_level_start_block_idx;
            }
        }

        inline int64_t get_left_child(int64_t node_idx) const override {
            // get block index
            int64_t block_idx = get_block_index(node_idx);
            if (block_idx == -1) return 2;
            // whether on the last row
            int64_t block_bottom_left_node_idx = get_last_row_first_element_inside_block(block_idx);
            if (node_idx < block_bottom_left_node_idx) {
                // not the last row.
                int64_t block_second_row_first_node_idx = get_second_row_first_element_inside_block(block_idx);
                int64_t offset = block_second_row_first_node_idx - 2;
                int64_t normalized_node_idx = node_idx - offset;
                return normalized_node_idx * 2 + offset;
            } else {
                // last row. the child is at next level block
                int64_t next_level_start_block_idx = m_block_branch_factor * block_idx + 1;
                int64_t next_level_block_idx = next_level_start_block_idx + node_idx - block_bottom_left_node_idx;
                return get_second_row_first_element_inside_block(next_level_block_idx);
            }
        }

        inline int64_t get_right_child(int64_t node_idx) const override {
            return get_left_child(node_idx) + 1;
        }

    private:
        int64_t m_partition_height;
        int64_t m_block_height;
        int64_t m_block_branch_factor;
        int64_t m_bottom_left_block_idx;

        // given a node index, return the block index (0-based)
        inline int64_t get_block_index(int64_t node_idx) const {
            if (node_idx == 1) return -1;
            return ((node_idx - 2) / (m_block_branch_factor - 1)) >> 1;
        }

        // given a block index, return the parent block index
        inline int64_t get_parent_block_index(int64_t block_idx) const {
            return (block_idx - 1) >> (m_partition_height - 1);
        }

        inline int64_t get_second_row_first_element_inside_block(int64_t block_idx) const {
            return block_idx * 2 * (m_block_branch_factor - 1) + 2;
        }

        inline int64_t get_last_row_first_element_inside_block(int64_t block_idx) const {
            return ((block_idx * (m_block_branch_factor - 1)) << 1) + m_block_branch_factor;
        }


        inline int64_t get_parent_block_idx(int64_t block_idx) const {
            return (block_idx - 1) >> (m_partition_height - 1);
        }
    };

}
#endif //HIPC21_SEGMENT_TREE_H
