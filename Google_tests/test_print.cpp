//
// Created by Chi Zhang on 7/31/21.
//

#include "gtest/gtest.h"
#include <iostream>

TEST(common, optional) {
//     the usage is very similar to Python None.
    fmt::print("Optional value is Nullopt\n");
    std::optional<float> val;
    if (val == std::nullopt) {
        fmt::print("Optional value is Nullopt\n");
    } else {
        fmt::print("Optional value is {}\n", val.value());
    }
    val = 12;
    if (val == std::nullopt) {
        fmt::print("Optional value is Nullopt\n");
    } else {
        fmt::print("Optional value is {}\n", val.value());
    }

}