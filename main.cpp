//
// Created by chi on 7/2/21.
//


#include "agent/dqn.h"
#include <iostream>


int main(int argc, char **argv) {
    try {
        dqn_main(argc, argv);
    } catch (const std::exception &e) {
        fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }

    return 0;
}