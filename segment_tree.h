//
// Created by chi on 7/3/21.
//

#ifndef HIPC21_SEGMENT_TREE_H

#include <torch/torch.h>

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

#define HIPC21_SEGMENT_TREE_H

#endif //HIPC21_SEGMENT_TREE_H
