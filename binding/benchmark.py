import numpy as np
import sumtree
import timeit

if __name__ == '__main__':
    capacity = [1000, 10000, 100000, 1000000, 10000000]
    K = [2, 4, 8, 16, 32, 64]
    batch_size = [32, 64, 128, 256, 512]
    iterations = 1000


    def insertion(tree, capacity, batch_size):
        idx = np.random.randint(capacity, size=batch_size)
        priority = np.random.rand(batch_size)
        tree.vector_set(idx, priority)


    def sample(tree: sumtree.SumTreefloat, batch_size):
        data = np.random.rand(batch_size) * tree.reduce()
        a = tree.vector_get_prefix_sum_idx(data)


    data = {'Capacity': [],
            "Batch Size": [],
            "K": [],
            "Insertion": [],
            "Sampling": []}

    for c in capacity:
        for k in K:
            tree = sumtree.SumTreefloat(c, k)
            for b in batch_size:
                insertion_time = timeit.timeit(lambda: insertion(tree, c, b), number=iterations) / iterations * 1000
                print(f'Capacity: {c}, k: {k}, batch size: {b}, runtime of insertion: {insertion_time}ms')
                sampling_time = timeit.timeit(lambda: sample(tree, b), number=iterations) / iterations * 1000
                print(f'Capacity: {c}, k: {k}, batch size: {b}, runtime of sampling: {sampling_time}ms')

                data["Capacity"].append(c)
                data["Batch Size"].append(b)
                data["K"].append(k)
                data["Insertion"].append(insertion_time)
                data["Sampling"].append(sampling_time)

    import pandas as pd

    dataframe = pd.DataFrame(data)