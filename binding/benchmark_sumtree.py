from ray.rllib.execution.segment_tree import SumSegmentTree
import numpy as np
import sumtree
import timeit
from tianshou.data.buffer.prio import SegmentTree


def insert_ray_tree(tree: SumSegmentTree, capacity, batch_size):
    idx = np.random.randint(capacity, size=batch_size)
    priority = np.random.rand(batch_size)
    for i, p in zip(idx, priority):
        tree[i] = p


def sample_ray_tree(tree: SumSegmentTree, batch_size):
    data = np.random.rand(batch_size) * tree.reduce()
    for d in data:
        tree.find_prefixsum_idx(d)


def insert_tianshou(tree: SegmentTree, capacity, batch_size):
    idx = np.random.randint(capacity, size=batch_size)
    priority = np.random.rand(batch_size)
    tree[idx] = priority


def sample_tianshou(tree: SegmentTree, batch_size):
    data = np.random.rand(batch_size) * tree.reduce()
    tree.get_prefix_sum_idx(data)


def insert_our(tree, capacity, batch_size):
    idx = np.random.randint(capacity, size=batch_size)
    priority = np.random.rand(batch_size)
    tree.vector_set(idx, priority)


def sample_our(tree: sumtree.SumTreefloat, batch_size):
    data = np.random.rand(batch_size) * tree.reduce()
    a = tree.vector_get_prefix_sum_idx(data)


if __name__ == '__main__':
    capacity = [1000, 10000, 100000, 1000000]
    K = [2, 4, 8, 16, 32, 64]
    batch_size = [128]
    iterations = 1000

    data = {'Capacity': [],
            "Batch Size": [],
            "Tree": [],
            "Insertion": [],
            "Sampling": []}

    tree_ray = SumSegmentTree
    tree_tianshou = SegmentTree
    tree_our = lambda c: sumtree.SumTreefloat(c, 16)

    insert_fn_dict = {
        tree_ray: insert_ray_tree,
        tree_tianshou: insert_tianshou,
        tree_our: insert_our
    }

    sample_fn_dict = {
        tree_ray: sample_ray_tree,
        tree_tianshou: sample_tianshou,
        tree_our: sample_our
    }

    tree_str_dict = {
        tree_ray: "rllib",
        tree_tianshou: "tianshou",
        tree_our: "ours"
    }

    for tree_fn in [tree_ray, tree_tianshou, tree_our]:

        for c in capacity:

            c = 2 ** (int(np.log2(c)) + 1)

            tree = tree_fn(c)
            for b in batch_size:
                insert_fn = insert_fn_dict[tree_fn]
                sample_fn = sample_fn_dict[tree_fn]
                tree_str = tree_str_dict[tree_fn]

                insertion_time = timeit.timeit(lambda: insert_fn(tree, c, b), number=iterations) / iterations * 1000
                print(f'Capacity: {c}, Tree: {tree_str}, batch size: {b}, runtime of insertion: {insertion_time}ms')
                sampling_time = timeit.timeit(lambda: sample_fn(tree, b), number=iterations) / iterations * 1000
                print(f'Capacity: {c}, Tree: {tree_str}, batch size: {b}, runtime of sampling: {sampling_time}ms')

                data["Capacity"].append(c)
                data["Batch Size"].append(b)
                data["Tree"].append(tree_str)
                data["Insertion"].append(insertion_time)
                data["Sampling"].append(sampling_time)

    import pandas as pd

    dataframe = pd.DataFrame(data)
    dataframe.to_csv("sumtree_library_com.csv", index=False)
