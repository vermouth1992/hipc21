#include <iostream>
#include <Eigen/Dense>


int main() {
    for (int i = 0; i < 10; i++) {
        Eigen::VectorXf m = Eigen::VectorXf::Random(3);
        std::cout << m << std::endl << std::endl;
    }
}
