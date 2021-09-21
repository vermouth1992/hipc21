//
// Created by Chi Zhang on 9/21/21.
//

#include "replay_buffer/segment_tree_cppopt.h"

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
