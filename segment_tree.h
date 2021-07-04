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

    std::unique_ptr<torch::Tensor> operator[](const torch::Tensor &idx) {
        return std::make_unique<torch::Tensor>(m_values.index({idx + m_bound}));
    }

    void set(const torch::Tensor &idx, const torch::Tensor &value) {
        torch::Tensor idx_bound = idx + m_bound;
        m_values.index_put_({idx_bound}, value);
        while (idx_bound.index({0}).item().toInt() > 1) {
            idx_bound = idx_bound / 2;
            auto left = m_values.index({idx_bound * 2});
            auto right = m_values.index({idx_bound * 2 + 1});
            m_values.index_put_({idx_bound}, left + right);
        }
    }

    double reduce(int64_t start, int64_t end) {
        return 0;
    }

    std::unique_ptr<torch::Tensor> get_prefix_sum_idx(const torch::Tensor &value) {

    }


private:
    int64_t m_size;
    int64_t m_bound;
    torch::Tensor m_values;
};

#define HIPC21_SEGMENT_TREE_H

#endif //HIPC21_SEGMENT_TREE_H
