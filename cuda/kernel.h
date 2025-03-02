#pragma once

#include <Eigen/Dense>

namespace mumpy::cuda {

Eigen::VectorXf vector_add(const Eigen::VectorXf& x, const Eigen::VectorXf& y);

using MatrixXfRowMajor = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
MatrixXfRowMajor matmul(const MatrixXfRowMajor& x, const MatrixXfRowMajor& y);

} // mumpy::cuda
