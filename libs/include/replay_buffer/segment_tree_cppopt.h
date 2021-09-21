//
// Created by Chi Zhang on 9/21/21.
//

#ifndef HIPC21_SEGMENT_TREE_CPPOPT_H
#define HIPC21_SEGMENT_TREE_CPPOPT_H


#include "segment_tree_cpp.h"

namespace rlu::replay_buffer {
    class SegmentTreeCPPOpt : public SegmentTreeCPP {
    public:
        explicit SegmentTreeCPPOpt(int64_t size, int64_t partition_height);

    protected:
        [[nodiscard]] int64_t convert_to_node_idx(int64_t data_idx) const override;

        [[nodiscard]] int64_t convert_to_data_idx(int64_t node_idx) const override;

        [[nodiscard]] int64_t get_parent(int64_t node_idx) const override;

        [[nodiscard]] int64_t get_left_child(int64_t node_idx) const override;

        [[nodiscard]] int64_t get_right_child(int64_t node_idx) const override;

    private:
        int64_t m_partition_height;
        int64_t m_block_height;
        int64_t m_block_branch_factor;
        int64_t m_bottom_left_block_idx;

        // given a node index, return the block index (0-based)
        [[nodiscard]] int64_t get_block_index(int64_t node_idx) const;

        // given a block index, return the parent block index
        [[nodiscard]] int64_t get_parent_block_index(int64_t block_idx) const;

        [[nodiscard]] int64_t get_second_row_first_element_inside_block(int64_t block_idx) const;

        [[nodiscard]] int64_t get_last_row_first_element_inside_block(int64_t block_idx) const;

        [[nodiscard]] int64_t get_parent_block_idx(int64_t block_idx) const;
    };

}


#endif //HIPC21_SEGMENT_TREE_CPPOPT_H
