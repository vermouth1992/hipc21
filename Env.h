//
// Created by Chi Zhang on 6/29/21.
//

#ifndef HIPC21_ENV_H
#define HIPC21_ENV_H

#include "Eigen/Dense"

class Env {

public:
    // step
    virtual Eigen::VectorXd step(Eigen::VectorXd action) = 0;
    // reset
    virtual Eigen::VectorXd reset() = 0;
    // set the seed of the environment
    virtual void seed(int seed) = 0;
};


#endif //HIPC21_ENV_H
