//
// Created by chi on 7/3/21.
//

#ifndef HIPC21_SEGMENT_TREE_H

#include <torch/torch.h>
#include <vector>
#include "omp.h"

class SegmentTree {
public:
    explicit SegmentTree(int64_t size) : m_size(size) {
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
    explicit SegmentTreeCPP(int64_t size) : m_size(size) {
        m_bound = 1;
        while (m_bound < size) {
            m_bound = m_bound * 2;
        }

        m_values = std::make_shared<std::vector<float>>(m_bound * 2, 0.0);
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
        return node_idx >= m_bound;
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
        return (*m_values)[node_idx];
    }

    virtual inline void set_value(int64_t node_idx, float value) {
        (*m_values)[node_idx] = value;
    }

    std::shared_ptr<torch::Tensor> operator[](const torch::Tensor &idx) const {
        auto idx_vector = convert_tensor_to_vector<int64_t>(idx);
        auto output = torch::zeros_like(idx, torch::TensorOptions().dtype(torch::kFloat32));
        for (int i = 0; i < idx_vector->size(); ++i) {
            output.index_put_({i}, get_value(convert_to_node_idx(idx_vector->at(i))));
        }
        return std::make_shared<torch::Tensor>(output);
    }

    void set(const torch::Tensor &idx, const torch::Tensor &value) {
        auto idx_vec = convert_tensor_to_vector<int64_t>(idx);
        auto value_vec = convert_tensor_to_vector<float>(value);
        // put all the values
        for (int i = 0; i < idx_vec->size(); ++i) {
            // get data pos
            int64_t pos = idx_vec->operator[](i);
            // get node pos
            pos = convert_to_node_idx(pos);
            // set the value of the leaf node
            set_value(pos, value_vec->operator[](i));
            // update the parent
            int64_t parent, sibling;
            while (pos != get_root()) {
                parent = get_parent(pos);
                sibling = get_sibling(pos);
                set_value(parent, get_value(pos) + get_value(sibling));
                pos = parent;
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

        for (int i = 0; i < value_vec->size(); i++) {
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

private:
    int64_t m_size;
    int64_t m_bound;
    std::shared_ptr<std::vector<float>> m_values;
};

#define HIPC21_SEGMENT_TREE_H

#endif //HIPC21_SEGMENT_TREE_H
