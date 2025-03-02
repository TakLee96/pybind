#include <iostream>

#include <cuda_runtime.h>

#include "kernel.h"

namespace mumpy::cuda {

#define CHECK(expr)                            \
  do {                                         \
    if (!(expr)) {                             \
      std::cerr << "CHECK failed: " << #expr;  \
      exit(1);                                 \
    }                                          \
  } while (0)

#define CUDA_CHECK(expr)                                                \
  do {                                                                  \
    cudaError_t err = (expr);                                           \
    if (err != cudaSuccess) {                                           \
      std::cerr << "CUDA_CHECK failed: " << #expr                       \
                << "\nCUDA Error Code" << err                           \
                << "\nError String:" << cudaGetErrorString(err);        \
      exit(err);                                                        \
    }                                                                   \
  } while (0)

__global__ void vector_add_kernel(const double *A, const double *B, double *C,
                                  int num_elements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < num_elements) {
    C[i] = A[i] + B[i];
  }
}

Eigen::VectorXd vector_add(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
  CHECK(x.size() == y.size());
  Eigen::VectorXd z;
  z.resizeLike(x);
  z.setZero();

  // Allocate the device inputs and outputs
  int num_elements = x.size();
  size_t size = num_elements * sizeof(double);
  double *d_A = nullptr;
  CUDA_CHECK(cudaMalloc((void **)&d_A, size));
  double *d_B = nullptr;
  CUDA_CHECK(cudaMalloc((void **)&d_B, size));
  double *d_C = nullptr;
  CUDA_CHECK(cudaMalloc((void **)&d_C, size));

  // Copy and launch
  CUDA_CHECK(cudaMemcpy(d_A, x.data(), size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, y.data(), size, cudaMemcpyHostToDevice));
  int threadsPerBlock = 256;
  int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
  vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, num_elements);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(z.data(), d_C, size, cudaMemcpyDeviceToHost));

  // Free device global memory
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  return z;
}

__global__ void matmul_kernel(const double *A, const double *B, double *C, int m, int n, int k) { 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0;
    if (col < k && row < m) {
      for (int i = 0; i < n; i++) {
        sum += A[row * n + i] * B[i * k + col];
      }
      C[row * k + col] = sum;
    }
}

MatrixXdRowMajor matmul(const MatrixXdRowMajor& x, const MatrixXdRowMajor& y) {
  CHECK(x.cols() == y.rows());
  int m = x.rows();
  int n = x.cols();
  int k = y.cols();
  MatrixXdRowMajor z(m, k);
  z.setZero();

  // Allocate the device inputs and outputs
  double *d_A = nullptr;
  CUDA_CHECK(cudaMalloc((void **)&d_A, x.size() * sizeof(double)));
  double *d_B = nullptr;
  CUDA_CHECK(cudaMalloc((void **)&d_B, y.size() * sizeof(double)));
  double *d_C = nullptr;
  CUDA_CHECK(cudaMalloc((void **)&d_C, z.size() * sizeof(double)));

  // Copy and launch
  CUDA_CHECK(cudaMemcpy(d_A, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, y.data(), y.size() * sizeof(double), cudaMemcpyHostToDevice));
  int block_size = 16;
  unsigned int grid_rows = (m + block_size - 1) / block_size;
  unsigned int grid_cols = (k + block_size - 1) / block_size;
  dim3 dimGrid(grid_cols, grid_rows);
  dim3 dimBlock(block_size, block_size);
  matmul_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, n, k);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(z.data(), d_C, z.size() * sizeof(double), cudaMemcpyDeviceToHost));

  // Free device global memory
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  return z;
}

} // mumpy::cuda
