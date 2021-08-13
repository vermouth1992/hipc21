//
// Created by chi on 7/10/21.
//

#include "gtest/gtest.h"
#include "utils/rl_functional.h"
#include "utils/stop_watcher.h"

#define RAW

TEST(Array, cache) {
    int size = 100000;

#ifdef RAW
    auto *a = new float[size];
    for (int i = 0; i < size; ++i) {
        a[i] = 1.0;
    }
#else
    auto a = std::make_shared<std::vector<float>>(size, 0.);
#endif
    // measure half access vs. sequential access.
    int64_t counter = 0;
    int64_t iterations = 100000;
    int64_t start_idx = size - 1;
    float result = 0.;

    rlu::watcher::StopWatcher half, seq;

    half.start();
    for (int i = 0; i < iterations; ++i) {
        counter = 0;
        while (start_idx > 1) {
#ifdef RAW
            result += a[start_idx];
#else
            result += a->at(start_idx);
#endif
            start_idx -= 100;
            counter += 1;
        }
        start_idx = size - 1;
    }
    half.stop();

    std::cout << counter << std::endl;

    float result1 = 0.;
    int64_t counter1 = 0;
    int64_t start_idx1 = size - 1;
    int interval = 2;
    seq.start();
    for (int i = 0; i < iterations; ++i) {
        counter1 = 0;
        for (int j = 0; j < counter * interval; j = j + interval) {
#ifdef RAW
            result1 += a[j];
#else
            result1 += a->at(j);
#endif
            start_idx1 -= 10;
            counter1 += 1;
        }
        start_idx1 = size - 1;
    }
    seq.stop();

    std::cout << counter << " " << counter1 << std::endl;
    std::cout << half.seconds() << " " << seq.seconds() << std::endl;
    std::cout << result << " " << result1 << " " << start_idx << start_idx1 << std::endl;
#ifdef RAW
    delete[]a;
#endif
}
