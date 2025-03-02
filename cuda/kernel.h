#pragma once

#include <Eigen/Dense>

namespace mumpy::cuda {

Eigen::VectorXd vector_add(const Eigen::VectorXd& x, const Eigen::VectorXd& y);

using MatrixXdRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
MatrixXdRowMajor matmul(const MatrixXdRowMajor& x, const MatrixXdRowMajor& y);

} // mumpy::cuda
