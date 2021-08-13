//
// Created by chi on 7/17/21.
//

#include "replay_buffer/segment_tree.h"

// SegmentTreeTorch

rlu::replay_buffer::SegmentTreeTorch::SegmentTreeTorch(int64_t size) : m_size(size) {
    m_bound = 1;
    while (m_bound < size) {
        m_bound = m_bound * 2;
    }
    m_values = torch::zeros({m_bound * 2}, torch::TensorOptions().dtype(torch::kFloat32));
}

int64_t rlu::replay_buffer::SegmentTreeTorch::size() const {
    return m_size;
}

torch::Tensor rlu::replay_buffer::SegmentTreeTorch::operator[](const torch::Tensor &idx) const {
    return m_values.index({idx + m_bound});
}

void rlu::replay_buffer::SegmentTreeTorch::set(const torch::Tensor &idx, const torch::Tensor &value) {
    torch::Tensor idx_bound = idx + m_bound;
    m_values.index_put_({idx_bound}, value);
    while (idx_bound.index({0}).item().toInt() > 1) {
        idx_bound = torch::div(idx_bound, 2, "floor");
        auto left = m_values.index({idx_bound * 2});
        auto right = m_values.index({idx_bound * 2 + 1});
        m_values.index_put_({idx_bound}, left + right);
    }
}

float rlu::replay_buffer::SegmentTreeTorch::reduce(int64_t start, int64_t end) const {
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

float rlu::replay_buffer::SegmentTreeTorch::reduce() const {
    return m_values.index({1}).item().toFloat();
}

torch::Tensor rlu::replay_buffer::SegmentTreeTorch::get_prefix_sum_idx(torch::Tensor value) const {
    auto index = torch::ones_like(value, torch::TensorOptions().dtype(torch::kInt64));
    while (index.index({0}).item().toInt() < m_bound) {
        index = index * 2;
        auto lsons = m_values.index({index});
        auto direct = torch::less(lsons, value);
        value = value - lsons * direct;
        index = index + direct;
    }
    index = index - m_bound;
    return index;
}


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

// SegmentTree Nary

rlu::replay_buffer::SegmentTreeNary::SegmentTreeNary(int64_t size, int64_t n) :
        m_n(n),
        m_size(size) {
    last_level_size = 1;
    while (last_level_size < size) {
        last_level_size = last_level_size * m_n;
    }
    m_bound = (last_level_size - 1) / (m_n - 1);
    initialize();
}

int64_t rlu::replay_buffer::SegmentTreeNary::size() const {
    return m_size;
}

int64_t rlu::replay_buffer::SegmentTreeNary::get_node_idx_after_padding(int64_t node_idx) const {
    return node_idx + m_padding;
}

float rlu::replay_buffer::SegmentTreeNary::get_value(int64_t node_idx) const {
    node_idx = get_node_idx_after_padding(node_idx);
    auto value = m_values[node_idx];
    return value;
}

void rlu::replay_buffer::SegmentTreeNary::set_value(int64_t node_idx, float value) {
    node_idx = get_node_idx_after_padding(node_idx);
    m_values[node_idx] = value;
}

int64_t rlu::replay_buffer::SegmentTreeNary::convert_to_node_idx(int64_t data_idx) const {
    return data_idx + m_bound;
}

int64_t rlu::replay_buffer::SegmentTreeNary::convert_to_data_idx(int64_t node_idx) const {
    return node_idx - m_bound;
}

int64_t rlu::replay_buffer::SegmentTreeNary::get_parent(int64_t node_idx) const {
    return (node_idx - 1) >> log2_m_n;
}

int64_t rlu::replay_buffer::SegmentTreeNary::get_root() const {
    return 0;
}

torch::Tensor rlu::replay_buffer::SegmentTreeNary::operator[](const torch::Tensor &idx) const {
    auto idx_vector = nn::convert_tensor_to_flat_vector<int64_t>(idx);
    auto output = torch::zeros_like(idx, torch::TensorOptions().dtype(torch::kFloat32));
    for (int i = 0; i < (int) idx_vector.size(); ++i) {
        output.index_put_({i}, get_value(convert_to_node_idx(idx_vector.at(i))));
    }
    return (output);
}

void rlu::replay_buffer::SegmentTreeNary::set(const torch::Tensor &idx, const torch::Tensor &value) {
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

float rlu::replay_buffer::SegmentTreeNary::reduce() const {
    return get_value(get_root());
}

float rlu::replay_buffer::SegmentTreeNary::reduce(int64_t start, int64_t end) const {
    assert(start >= 0 && end <= size() && end >= start);
    if (start == 0) {
        return reduce(end);
    } else return reduce(end) - reduce(start);
}

float rlu::replay_buffer::SegmentTreeNary::reduce(int64_t end) const {
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

torch::Tensor rlu::replay_buffer::SegmentTreeNary::get_prefix_sum_idx(torch::Tensor value) const {
    auto value_vec = nn::convert_tensor_to_flat_vector<float>(value);
    auto index = torch::ones_like(value, torch::TensorOptions().dtype(torch::kInt64));

    for (int i = 0; i < (int) value_vec.size(); i++) {
        int64_t idx = get_root();
        float current_val = value_vec[i];
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
        index.index_put_({i}, convert_to_data_idx(idx));
    }

    return index;
}

bool rlu::replay_buffer::SegmentTreeNary::is_leaf(int64_t node_idx) const {
    return node_idx >= m_bound;
}

int64_t rlu::replay_buffer::SegmentTreeNary::get_left_child(int64_t node_idx) const {
    // using shift operator is crucial
    return (node_idx << log2_m_n) + 1;
}

void rlu::replay_buffer::SegmentTreeNary::initialize() {
    // zero-based indexing
    int64_t total_size = (last_level_size * m_n - 1) / (m_n - 1);
    // making the data at each level cache aligned
    m_padding = m_n - 1;
    log2_m_n = (int64_t) std::log2(m_n);
    m_values = new float[total_size + m_padding];
    for (int i = 0; i < total_size; ++i) {
        m_values[i] = 0.;
    }
//    spdlog::info("SegmentTreeNary, n = {0}, size = {1}, m_bound = {2}", m_n, m_size, m_bound);
}

// SegmentTreeCPPOpt

rlu::replay_buffer::SegmentTreeCPPOpt::SegmentTreeCPPOpt(int64_t size, int64_t partition_height) :
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

int64_t rlu::replay_buffer::SegmentTreeCPPOpt::convert_to_node_idx(int64_t data_idx) const {
    // get the block offset towards the bottom left block
    int64_t block_offset = data_idx >> (m_partition_height - 1);
    int64_t block_idx = block_offset + m_bottom_left_block_idx;
    // get the index offset
    int64_t index_offset = data_idx % m_block_branch_factor;
    // compute the index
    int64_t block_bottom_left_node_idx = get_last_row_first_element_inside_block(block_idx);
    return block_bottom_left_node_idx + index_offset;
}

int64_t rlu::replay_buffer::SegmentTreeCPPOpt::convert_to_data_idx(int64_t node_idx) const {
    // node_idx must be leaf node
    // compute block index
    int64_t block_idx = get_block_index(node_idx);
    int64_t block_offset = block_idx - m_bottom_left_block_idx;
    int64_t block_bottom_left_node_idx = get_last_row_first_element_inside_block(block_idx);
    int64_t index_offset = node_idx - block_bottom_left_node_idx;
    // compute offset index
    return block_offset * m_block_branch_factor + index_offset;
}

int64_t rlu::replay_buffer::SegmentTreeCPPOpt::get_parent(int64_t node_idx) const {
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

int64_t rlu::replay_buffer::SegmentTreeCPPOpt::get_left_child(int64_t node_idx) const {
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

int64_t rlu::replay_buffer::SegmentTreeCPPOpt::get_right_child(int64_t node_idx) const {
    return get_left_child(node_idx) + 1;
}

int64_t rlu::replay_buffer::SegmentTreeCPPOpt::get_block_index(int64_t node_idx) const {
    if (node_idx == 1) return -1;
    return ((node_idx - 2) / (m_block_branch_factor - 1)) >> 1;
}

int64_t rlu::replay_buffer::SegmentTreeCPPOpt::get_parent_block_index(int64_t block_idx) const {
    return (block_idx - 1) >> (m_partition_height - 1);
}

int64_t rlu::replay_buffer::SegmentTreeCPPOpt::get_second_row_first_element_inside_block(int64_t block_idx) const {
    return block_idx * 2 * (m_block_branch_factor - 1) + 2;
}

int64_t rlu::replay_buffer::SegmentTreeCPPOpt::get_last_row_first_element_inside_block(int64_t block_idx) const {
    return ((block_idx * (m_block_branch_factor - 1)) << 1) + m_block_branch_factor;
}

int64_t rlu::replay_buffer::SegmentTreeCPPOpt::get_parent_block_idx(int64_t block_idx) const {
    return (block_idx - 1) >> (m_partition_height - 1);
}
