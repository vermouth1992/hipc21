//
// Created by Chi Zhang on 6/29/21.
//

#ifndef HIPC21_CARTPOLE_H
#define HIPC21_CARTPOLE_H

#include <string>
#include "Env.h"
#include <cmath>

class CartPole : Env {
private:
    const float gravity = 9.8;
    const float masscart = 1.0;
    const float masspole = 0.1;
    const float total_mass = masspole + masscart;
    const float length = 0.5;
    const float polemass_length = masspole * length;
    const float force_mag = 10.0;
    const float tau = 0.02;
    const std::string kinematics_integrator = "euler";
    const float theta_threshold_radians = 12 * 2 * M_PI / 360;
    const float x_threshold = 2.4;
    Eigen::VectorXd state;

public:
    Eigen::VectorXd step(Eigen::VectorXd action) override;

    Eigen::VectorXd reset() override;

    void seed(int seed) override;
};


#endif //HIPC21_CARTPOLE_H
