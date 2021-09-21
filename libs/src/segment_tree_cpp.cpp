//
// Created by Chi Zhang on 9/21/21.
//

#include "replay_buffer/segment_tree_cpp.h"

// SegmentTree C++

rlu::replay_buffer::SegmentTreeCPP::~SegmentTreeCPP() {
    delete[]m_values;
}

rlu::replay_buffer::SegmentTreeCPP::SegmentTreeCPP(int64_t size) : m_size(size) {
    m_bound = 1;
    while (m_bound < size) {
        m_bound = m_bound * 2;
    }
    initialize();
}

int64_t rlu::replay_buffer::SegmentTreeCPP::size() const {
    return m_size;
}

int64_t rlu::replay_buffer::SegmentTreeCPP::convert_to_node_idx(int64_t data_idx) const {
    return data_idx + m_bound;
}

int64_t rlu::replay_buffer::SegmentTreeCPP::convert_to_data_idx(int64_t node_idx) const {
    return node_idx - m_bound;
}

int64_t rlu::replay_buffer::SegmentTreeCPP::get_parent(int64_t node_idx) const {
    return node_idx / 2;
}

int64_t rlu::replay_buffer::SegmentTreeCPP::get_sibling(int64_t node_idx) const {
    if (node_idx % 2 == 0) {
        return node_idx + 1;
    } else {
        return node_idx - 1;
    }
}

bool rlu::replay_buffer::SegmentTreeCPP::is_leaf(int64_t node_idx) const {
    int64_t left_child = get_left_child(node_idx);
    return left_child >= 2 * m_bound;
}

bool rlu::replay_buffer::SegmentTreeCPP::is_left(int64_t node_idx) const {
    return node_idx % 2 == 0;
}

torch::Tensor rlu::replay_buffer::SegmentTreeCPP::operator[](const torch::Tensor &idx) const {
    auto idx_vector = nn::convert_tensor_to_flat_vector<int64_t>(idx);
    auto output = torch::zeros_like(idx, torch::TensorOptions().dtype(torch::kFloat32));
    for (int i = 0; i < (int) idx_vector.size(); ++i) {
        output.index_put_({i}, get_value(convert_to_node_idx(idx_vector.at(i))));
    }
    return output;
}

void rlu::replay_buffer::SegmentTreeCPP::set(const torch::Tensor &idx, const torch::Tensor &value) {
    auto idx_vec = nn::convert_tensor_to_flat_vector<int64_t>(idx);
    auto value_vec = nn::convert_tensor_to_flat_vector<float>(value);
    // put all the values
    for (int i = 0; i < (int) idx_vec.size(); ++i) {
        // get data pos
        int64_t pos = idx_vec.operator[](i);
        // get node pos
        pos = convert_to_node_idx(pos);
        // set the value of the leaf node
        auto original_value = get_value(pos);
        auto new_value = value_vec.operator[](i);
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

float rlu::replay_buffer::SegmentTreeCPP::reduce(int64_t start, int64_t end) const {
    assert(start >= 0 && end <= size() && end >= start);
    if (start == 0) {
        return reduce(end);
    } else return reduce(end) - reduce(start);
}

float rlu::replay_buffer::SegmentTreeCPP::reduce(int64_t end) const {
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

torch::Tensor rlu::replay_buffer::SegmentTreeCPP::get_prefix_sum_idx(torch::Tensor value) const {
    auto value_vec = nn::convert_tensor_to_flat_vector<float>(value);
    auto index = torch::ones_like(value, torch::TensorOptions().dtype(torch::kInt64));

    for (int i = 0; i < (int) value_vec.size(); i++) {
        int64_t idx = get_root();
        float current_val = value_vec[i];
        while (!is_leaf(idx)) {
            idx = get_left_child(idx);
            auto lsons = get_value(idx);
            if (lsons < current_val) {
                current_val -= lsons;
                idx = get_sibling(idx);
            }
        }
        index.index_put_({i}, convert_to_data_idx(idx));
    }

    return index;
}

bool rlu::replay_buffer::SegmentTreeCPP::is_right(int64_t node_idx) const {
    return node_idx % 2 == 1;
}

int64_t rlu::replay_buffer::SegmentTreeCPP::get_left_child(int64_t node_idx) const {
    return node_idx * 2;
}

int64_t rlu::replay_buffer::SegmentTreeCPP::get_right_child(int64_t node_idx) const {
    return node_idx * 2 + 1;
}

int64_t rlu::replay_buffer::SegmentTreeCPP::get_root() const {
    return 1;
}

float rlu::replay_buffer::SegmentTreeCPP::get_value(int64_t node_idx) const {
    auto value = m_values[node_idx];
    return value;
}

void rlu::replay_buffer::SegmentTreeCPP::set_value(int64_t node_idx, float value) {
    m_values[node_idx] = value;
}

float rlu::replay_buffer::SegmentTreeCPP::reduce() const {
    return get_value(get_root());
}

void rlu::replay_buffer::SegmentTreeCPP::initialize() {
//    spdlog::info("SegmentTreeCPP m_bound = {}", m_bound);
    m_values = new float[m_bound * 2];
    for (int i = 0; i < m_bound * 2; ++i) {
        m_values[i] = 0.;
    }
}