//
// Created by Chi Zhang on 8/1/21.
//

#include "gtest/gtest.h"
#include "Python/Python.h"

TEST(python, basic) {
    Py_Initialize();
    PyRun_SimpleString("import gym, time\n"
                       "env = gym.make('CartPole-v1')\n"
                       "env.reset()\n"
                       "for i in range(10):\n"
                       "    time.sleep(0.5)\n"
                       "    env.step(env.action_space.sample())\n"
                       "    env.render()\n");
}