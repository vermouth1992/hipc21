//
// Created by Chi Zhang on 7/31/21.
//

#include "gtest/gtest.h"
#include <iostream>
#include "fmt/format.h"
#include "utils/cpp_functional.h"

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

TEST(common, get_default) {
    std::unordered_map<std::string, std::any> notebook;

    std::string name{"Pluto"};
    int year = 2015;

    notebook["PetName"] = name;
    notebook["Born"] = year;

    auto name2 = std::any_cast<std::string>(notebook["PetName"]); // = "Pluto"
    auto year2 = std::any_cast<int>(notebook["Born"]); // = 2015

    ASSERT_EQ(name2, "Pluto");
    ASSERT_EQ(year2, 2015);

    auto name3 = rlu::cpp_functional::get_any_with_default_val<std::string, std::string>(notebook, "Born1", "dss");
    auto year3 = rlu::cpp_functional::get_any_with_default_val<std::string, int>(notebook, "Born", 2014);
    ASSERT_EQ(name3, "dss");
    ASSERT_EQ(year3, 2015);
}