//
// Created by chi on 7/5/21.
//

#include <gtest/gtest.h>
#include <vector>

struct C {
    C() = default;

    C(const C &) { std::cout << "A copy was made.\n"; }
};

C f() {
    return C();
}


TEST(shared_ptr, copy) {
    auto *a = new std::vector<int>{1, 2, 3};
    std::cout << "Address of a is " << a << std::endl;

    // create a copy of a...
    std::shared_ptr<std::vector<int>> ptr1 = std::make_shared<std::vector<int>>(*a);
    std::cout << ptr1.get() << std::endl;

    // point to the same a...
    std::shared_ptr<std::vector<int>> ptr2(a);
    std::cout << ptr2.get() << std::endl;


    std::cout << "Hello World!\n";
    C obj = f();

}