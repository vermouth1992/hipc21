//
// Created by chi on 7/17/21.
//

#ifndef HIPC21_SUMTREE_H
#define HIPC21_SUMTREE_H

#include <cstdint>

// N-ary sum tree
template<class T>
class SumTree {
public:
    explicit SumTree(int64_t size, int64_t n);

    ~SumTree();

    int64_t size() const;

    T operator[](int64_t idx) const;

    void set(int64_t idx, T value);

    T reduce() const;

    T reduce(int64_t start, int64_t end) const;

    T reduce(int64_t end) const;

    int64_t get_prefix_sum_idx(T value) const;

protected:
    int64_t m_n{};
    int64_t log2_m_n{};
    int64_t m_size{};
    // the size of the last level
    int64_t last_level_size{};
    int64_t m_bound{};
    int64_t m_padding{};
    T *m_values{};

    virtual inline int64_t get_node_idx_after_padding(int64_t node_idx) const;

    virtual inline T get_value(int64_t node_idx) const;

    virtual inline void set_value(int64_t node_idx, T value);

    virtual inline int64_t convert_to_node_idx(int64_t data_idx) const;

    virtual inline int64_t convert_to_data_idx(int64_t node_idx) const;

    virtual inline int64_t get_parent(int64_t node_idx) const;

    virtual inline int64_t get_root() const;

    virtual inline bool is_leaf(int64_t node_idx) const;

    virtual inline int64_t get_left_child(int64_t node_idx) const;


};


#endif //HIPC21_SUMTREE_H
