#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

namespace mumpy::linalg {

// EQUIVALENT: using MatrixXdGeneric = pybind11::EigenDRef<Eigen::MatrixXd>;
using MatrixXdGeneric = Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
using MatrixXdRow = Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
using MatrixXdCol = Eigen::Ref<Eigen::MatrixXd>;

Eigen::MatrixXd matmul_generic(const MatrixXdGeneric& x, const MatrixXdGeneric& y);

Eigen::MatrixXd matmul_row(const MatrixXdRow& x, const MatrixXdRow& y);

Eigen::MatrixXd matmul_col(const MatrixXdCol& x, const MatrixXdCol& y);

} // namespace mumpy::linalg