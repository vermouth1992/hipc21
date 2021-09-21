//
// Created by chi on 7/17/21.
//

#include "replay_buffer/segment_tree_base.h"


auto rlu::replay_buffer::SegmentTree::sample_idx(int64_t batch_size) const -> torch::Tensor {
    auto scalar = torch::rand({batch_size}) * reduce();
    auto idx = this->get_prefix_sum_idx(scalar);
    return idx;
}
