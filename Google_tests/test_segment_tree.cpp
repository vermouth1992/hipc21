//
// Created by chi on 7/4/21.
//

#include "gtest/gtest.h"
#include "segment_tree.h"
#include "functional.h"

TEST(SegmentTree, speed) {
    StopWatcher origin;
    StopWatcher optimized;
    int64_t tree_size = 1000000;
    int64_t batch_size = 256;
    SegmentTreeCPPOpt tree_opt(tree_size, 3);
    SegmentTreeCPP tree(tree_size);
    int64_t iterations = 10000;
    // random set

    torch::manual_seed(1);
    origin.start();
    for (int i = 0; i < iterations; ++i) {
        auto idx = torch::randint(tree_size, {batch_size}, torch::TensorOptions().dtype(torch::kInt64));
        auto value = torch::rand({batch_size}) * 10;
        tree.set(idx, value);
    }
    origin.stop();

    torch::manual_seed(1);
    optimized.start();
    for (int i = 0; i < iterations; ++i) {
        auto idx = torch::randint(tree_size, {batch_size}, torch::TensorOptions().dtype(torch::kInt64));
        auto value = torch::rand({batch_size}) * 10;
        tree_opt.set(idx, value);
    }
    optimized.stop();


    std::cout << origin.seconds() << " " << optimized.seconds() << std::endl;


}

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

TEST(SegmentTreeCPPOpt, all) {
    int64_t tree_size = 10;
    SegmentTreeCPPOpt tree(tree_size, 4);
    torch::Tensor index = torch::arange(tree_size);
    std::cout << index << std::endl;
    torch::Tensor value = torch::rand({tree_size}) * 10;
    tree.set(index, value);
    auto value_vec = convert_tensor_to_vector<float>(value);
    std::cout << *value_vec << std::endl << value;
    std::cout << *(tree.operator[](torch::arange(tree_size)));

    for (int i = 1; i <= tree_size; ++i) {
        float true_sum = 0;
        for (int j = 0; j < i; ++j) {
            true_sum += value_vec->at(j);
        }
        std::cout << true_sum << " " << tree.reduce(i) << std::endl;
        ASSERT_NEAR(true_sum, tree.reduce(i), 1e-5);
    }


    ASSERT_NEAR(tree.reduce(), tree.reduce(0, tree_size), 1e-5);

    torch::Tensor randnum = torch::rand({1000}) * tree.reduce();
    auto sample_indexes = convert_tensor_to_vector<int64_t>(*tree.get_prefix_sum_idx(randnum));
    std::vector<float> frequency(tree_size, 0.);
    for (auto &i : *sample_indexes) {
        frequency[i] += 1;
    }
    auto max_freq = *std::max_element(frequency.begin(), frequency.end());
    for (float &i : frequency) {
        i /= max_freq;
    }
    max_freq = *std::max_element(value_vec->begin(), value_vec->end());
    for (float &i : *value_vec) {
        i /= max_freq;
    }
    std::cout << frequency << std::endl << *value_vec << std::endl;
}

TEST(SegmentTreeCPP, reduce) {
    int64_t tree_size = 10;
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
        ASSERT_NEAR(true_sum, tree.reduce(i), 1e-4);
    }


    ASSERT_NEAR(tree.reduce(), tree.reduce(0, tree_size), 1e-5);
    // leaf node
    std::cout << *(tree.operator[](torch::arange(tree_size)));
    // random access frequency
    torch::Tensor randnum = torch::rand({1000}) * tree.reduce();
    auto sample_indexes = convert_tensor_to_vector<int64_t>(*tree.get_prefix_sum_idx(randnum));
    std::vector<float> frequency(tree_size, 0.);
    for (auto &i : *sample_indexes) {
        frequency[i] += 1;
    }
    auto max_freq = *std::max_element(frequency.begin(), frequency.end());
    for (float &i : frequency) {
        i /= max_freq;
    }
    max_freq = *std::max_element(value_vec->begin(), value_vec->end());
    for (float &i : *value_vec) {
        i /= max_freq;
    }
    std::cout << frequency << std::endl << *value_vec << std::endl;

}