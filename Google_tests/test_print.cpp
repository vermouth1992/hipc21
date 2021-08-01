//
// Created by Chi Zhang on 7/31/21.
//

#include "gtest/gtest.h"
#include "logger.h"
#include <iostream>
#include <fmt/core.h>

TEST(print, color) {
    printf("\x1b[%sm%s\x1b[0m\n", "32", "dsdsd");
    std::vector<std::string> a{"123", "345"};
    std::string x = string_join(a, std::string(","));
    printf("\x1b[%sm%s\x1b[0m", "32", x.c_str());
    std::cout << colorize("321", "red", false, false) << std::endl;
    fmt::print("The answer is {}.\n", 42);
    fmt::print("{:>10}", "dds");
    fmt::print("{{:>{}}}", 10);
}

TEST(logger, main) {
    auto data_dir = "data";
    auto exp_name = "Hopper-v2_SAC";
    int seed = 1;
    auto output_dir = setup_logger_kwargs(exp_name, seed, data_dir);
    Logger logger(output_dir, exp_name);
    for (int i = 1; i < 100; i++) {
        logger.log_tabular("Epoch", i);
        float performance = i + (rand() % 20) / 19;
        logger.log_tabular("AverageEpRet", performance);
        logger.dump_tabular();
    }
}