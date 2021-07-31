//
// Created by chi on 7/16/21.
//

#include "gtest/gtest.h"
#include <cstdio>

#ifdef __APPLE__

#include <sys/sysctl.h>

#else
#include <sys/sysinfo.h>
#endif

#include <common.h>

TEST(cpu, core) {
#ifdef __APPLE__
    printf("This system has %ld processors configured and "
           "%ld processors available.\n",
           sysconf(_SC_NPROCESSORS_ONLN), sysconf(_SC_NPROCESSORS_ONLN));
#else
    printf("This system has %d processors configured and "
           "%d processors available.\n",
           get_nprocs_conf(), get_nprocs());
#endif
    M_Assert(false, "It is supposed to be false");
}