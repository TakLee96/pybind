import mumpy as mp
import numpy as np

print(f"{mp.add(2, 3)=}")

x = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
], dtype=np.float64)
y = np.array([
    [ 1],
    [ 0],
    [-1],
], dtype=np.float64)

print(f"{mp.matmul_generic(x, y)=}")
print(f"{mp.matmul_cuda(x, y)=}")
print(f"{x @ y =}")

print(f"{mp.vector_add_cuda(y, y)=}")
print(f"{y + y =}")
