//
// Created by chi on 7/17/21.
//

#include "SumTree.h"
#include "common.h"
#include <cmath>


constexpr bool is_power_of2(int v) {
    return v && ((v & (v - 1)) == 0);
}

template<class T>
SumTree<T>::SumTree(int64_t size, int64_t n):
        m_n(n),
        m_size(size) {
    // find the size of the last level
    M_Assert(is_power_of2(n), "n must be power of 2.");
    last_level_size = 1;
    while (last_level_size < size) {
        last_level_size = last_level_size * m_n;
    }
    m_bound = (last_level_size - 1) / (m_n - 1);
    // zero-based indexing
    int64_t total_size = (last_level_size * m_n - 1) / (m_n - 1);
    // making the data at each level cache aligned
    m_padding = m_n - 1;
    log2_m_n = (int64_t) std::log2(m_n);
    m_values = new T[total_size + m_padding];
    for (int i = 0; i < total_size; ++i) {
        m_values[i] = 0.;
    }
}

template<class T>
SumTree<T>::~SumTree() {
    delete[]m_values;
}

template<class T>
int64_t SumTree<T>::size() const {
    return m_size;
}

template<class T>
T SumTree<T>::operator[](int64_t idx) const {
    M_Assert(idx < size(), "idx is out of bound");
    return get_value(convert_to_node_idx(idx));
}

template<class T>
void SumTree<T>::set(int64_t idx, T value) {
    M_Assert(idx < size(), "idx is out of bound");
    int64_t pos = idx;
    // get node pos
    pos = convert_to_node_idx(pos);
    // set the value of the leaf node
    auto original_value = get_value(pos);
    auto new_value = value;
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

template<class T>
T SumTree<T>::reduce() const {
    return get_value(get_root());
}

template<class T>
T SumTree<T>::reduce(int64_t start, int64_t end) const {
    M_Assert(start >= 0 && end <= size() && end >= start, "invalid start and end");
    if (start == 0) {
        return reduce(end);
    } else return reduce(end) - reduce(start);
}

template<class T>
T SumTree<T>::reduce(int64_t end) const {
    M_Assert(end > 0 && end <= size(), "end is out of bound.");
    if (end == size()) {
        return reduce();
    }
    end = convert_to_node_idx(end);
    T result = 0.;
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

template<class T>
int64_t SumTree<T>::get_prefix_sum_idx(T value) const {
    int64_t idx = get_root();
    auto current_val = value;
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
    return convert_to_data_idx(idx);
}

template<class T>
int64_t SumTree<T>::get_node_idx_after_padding(int64_t node_idx) const {
    return node_idx + m_padding;
}

template<class T>
T SumTree<T>::get_value(int64_t node_idx) const {
    node_idx = get_node_idx_after_padding(node_idx);
    auto value = m_values[node_idx];
    return value;
}

template<class T>
void SumTree<T>::set_value(int64_t node_idx, T value) {
    node_idx = get_node_idx_after_padding(node_idx);
    m_values[node_idx] = value;
}

template<class T>
int64_t SumTree<T>::convert_to_node_idx(int64_t data_idx) const {
    return data_idx + m_bound;
}

template<class T>
int64_t SumTree<T>::convert_to_data_idx(int64_t node_idx) const {
    return node_idx - m_bound;
}

template<class T>
int64_t SumTree<T>::get_parent(int64_t node_idx) const {
    return (node_idx - 1) >> log2_m_n;
}

template<class T>
int64_t SumTree<T>::get_root() const {
    return 0;
}

template<class T>
bool SumTree<T>::is_leaf(int64_t node_idx) const {
    return node_idx >= m_bound;
}

template<class T>
int64_t SumTree<T>::get_left_child(int64_t node_idx) const {
    return (node_idx << log2_m_n) + 1;
}
