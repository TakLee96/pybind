#pragma once

#include <Eigen/Dense>

namespace mumpy::cuda {

Eigen::VectorXf vector_add(const Eigen::VectorXf& x, const Eigen::VectorXf& y);

} // mumpy::cuda
