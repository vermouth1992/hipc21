//
// Created by chi on 7/4/21.
//

#include "gtest/gtest.h"
#include "segment_tree.h"

TEST(SegmentTree, reduce) {
    int64_t tree_size = 20;
    int64_t index_size = 3;
    SegmentTree tree(tree_size);
    torch::Tensor index = torch::randint(tree_size, {index_size}, torch::TensorOptions().dtype(torch::kInt64));
    std::cout << index << std::endl;
    torch::Tensor value = torch::rand({index_size}) * 10;
    tree.set(index, value);
    ASSERT_NEAR(tree.reduce(), tree.reduce(0, tree_size), 1e-6);
    torch::Tensor randnum = torch::rand({3}) * tree.reduce();
    std::cout << *tree.get_prefix_sum_idx(randnum);
    std::cout << *(tree.operator[](torch::arange(tree_size)));
}

TEST(SegmentTreeCPP, reduce) {
    int64_t tree_size = 8;
    SegmentTreeCPP tree(tree_size);
    torch::Tensor index = torch::arange(tree_size);
    std::cout << index << std::endl;
    torch::Tensor value = torch::rand({tree_size}) * 10;
    tree.set(index, value);
    auto value_vec = convert_tensor_to_vector<float>(value);
    std::cout << *value_vec << std::endl << value;
    for (int i = 1; i <= tree_size; ++i) {
        float true_sum = 0;
        for (int j = 0; j < i; ++j) {
            true_sum += value_vec->at(j);
        }
        std::cout << true_sum << " " << tree.reduce(i) << std::endl;
        ASSERT_NEAR(true_sum, tree.reduce(i), 1e-5);
    }


    ASSERT_NEAR(tree.reduce(), tree.reduce(0, tree_size), 1e-5);
    torch::Tensor randnum = torch::rand({3}) * tree.reduce();
    std::cout << *tree.get_prefix_sum_idx(randnum);
    std::cout << *(tree.operator[](torch::arange(tree_size)));
}