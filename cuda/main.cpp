#include <iostream>

#include <Eigen/Dense>

#include "kernel.h"

int main(void) {
  Eigen::Vector3f x{ 0, 1.5,  2};
  Eigen::Vector3f y{ 0, 0.2, -3};
  Eigen::Vector3f z = mumpy::cuda::vector_add(x, y);

  Eigen::IOFormat heavy(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
  std::cout << x.format(heavy)
            << "\n+\n" << y.format(heavy)
            << "\n=\n" << z.format(heavy);
}
