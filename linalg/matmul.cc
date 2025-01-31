#include "matmul.h"

namespace mumpy::linalg {

Eigen::MatrixXd matmul_generic(const MatrixXdGeneric& x, const MatrixXdGeneric& y) {
    return x * y;
}

Eigen::MatrixXd matmul_row(const MatrixXdRow& x, const MatrixXdRow& y) {
    return x * y;
}

Eigen::MatrixXd matmul_col(const MatrixXdCol& x, const MatrixXdCol& y) {
    return x * y;
}

} // namespace mumpy::linalg
