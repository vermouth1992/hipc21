//
// Created by chi on 7/16/21.
//

#include "gtest/gtest.h"
#include <stdio.h>
#include <sys/sysinfo.h>


TEST(cpu, core) {
    printf("This system has %d processors configured and "
           "%d processors available.\n",
           get_nprocs_conf(), get_nprocs());

}