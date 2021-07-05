//
// Created by chi on 7/4/21.
//

#include "gtest/gtest.h"
#include "segment_tree.h"

TEST(SegmentTree, reduce) {
    int64_t tree_size = 20;
    int64_t index_size = 20;
    SegmentTree tree(tree_size);
    torch::Tensor index = torch::randint(tree_size, {index_size}, torch::TensorOptions().dtype(torch::kInt64));
    std::cout << index << std::endl;
    torch::Tensor value = torch::rand({index_size}) * 10;
    tree.set(index, value);
    ASSERT_NEAR(tree.reduce(), tree.reduce(0, tree_size), 1e-6);
    torch::Tensor randnum = torch::rand({3}) * tree.reduce();
    std::cout << *tree.get_prefix_sum_idx(randnum);
}