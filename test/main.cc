#include "simd_helper/simd_helper.h"

#include "Eigen/Dense"

int main(int argc, char** argv) {
  simd::Vector<3> v({Eigen::Vector3f(1, 2, 3), Eigen::Vector3f(1, 2, 3),
                     Eigen::Vector3f(1, 2, 3), Eigen::Vector3f(1, 2, 3),
                     Eigen::Vector3f(1, 2, 3), Eigen::Vector3f(1, 2, 3),
                     Eigen::Vector3f(4, 5, 6), Eigen::Vector3f(7, 8, 9)});

  std::cerr << v(2) << std::endl;

  return 0;
}