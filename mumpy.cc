#include <pybind11/pybind11.h>

#include "linalg/matmul.h"
#include "cuda/kernel.h"

namespace mumpy {

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(mumpy, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "add two integers");

    m.def("matmul_generic", &linalg::matmul_generic, "matmul any numpy array or slice");

    m.def("matmul_row", &linalg::matmul_row, "matmul only numpy array");

    m.def("matmul_col", &linalg::matmul_col, "matmul only col-major numpy array");

    m.def("vector_add_cuda", &cuda::vector_add, "add two vectors using cuda");
}

} // namespace mumpy
