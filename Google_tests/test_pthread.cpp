//
// Created by Chi Zhang on 8/9/21.
//

#include "gtest/gtest.h"
#include "pthread.h"

void *print_message_function(void *ptr);

TEST(pthread, pthread_multiple) {
    pthread_t thread1;
    char *message1 = "Thread 1";
    char *message2 = "Thread 2";
    int iret1, iret2;

    /* Create independent threads each of which will execute function */

    iret1 = pthread_create(&thread1, NULL, print_message_function, (void *) message1);
    pthread_join(thread1, NULL);
    iret2 = pthread_create(&thread1, NULL, print_message_function, (void *) message2);
    pthread_join(thread1, NULL);

    printf("Thread 1 returns: %d\n", iret1);
    printf("Thread 2 returns: %d\n", iret2);
}

void *print_message_function(void *ptr) {
    char *message;
    message = (char *) ptr;
    printf("%s \n", message);
}