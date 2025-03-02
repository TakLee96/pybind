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

__global__ void vector_add_kernel(const float *A, const float *B, float *C,
                                  int num_elements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < num_elements) {
    C[i] = A[i] + B[i];
  }
}

Eigen::VectorXf vector_add(const Eigen::VectorXf& x, const Eigen::VectorXf& y) {
  CHECK(x.size() == y.size());
  Eigen::VectorXf z;
  z.resizeLike(x);
  z.setZero();

  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;
  int num_elements = x.size();
  size_t size = num_elements * sizeof(float);

  // Allocate the device inputs and outputs
  float *d_A = NULL;
  CUDA_CHECK(cudaMalloc((void **)&d_A, size));
  float *d_B = NULL;
  CUDA_CHECK(cudaMalloc((void **)&d_B, size));
  float *d_C = NULL;
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

} // mumpy::cuda
