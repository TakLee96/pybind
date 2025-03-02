import mumpy as mp
import numpy as np
from time import time


class Timer:
    def __init__(self, name):
        self.name = name
    def __enter__(self, *args, **kwargs):
        self.t = time()
    def __exit__(self, *args, **kwargs):
        self.t = time() - self.t
        print(f"{self.name} elapsed: {self.t:.02f} sec")


def main():
    x = np.random.normal(size=(1024, 1024))
    y = np.random.normal(size=(1024, 1024))

    with Timer('numpy'):
        z_np = x @ y

    with Timer('mumpy.matmul_generic'):
        z_mp_cpu_gen = mp.matmul_generic(x, y)
    print(f"{(z_mp_cpu_gen - z_np).mean()=}")

    with Timer('mumpy.matmul_row'):
        z_mp_cpu_row = mp.matmul_row(x, y)
    print(f"{(z_mp_cpu_row - z_np).mean()=}")

    x_col = np.array(x, order='F')
    y_col = np.array(y, order='F')
    with Timer('mumpy.matmul_col'):
        z_mp_cpu_col = mp.matmul_col(x_col, y_col)
    print(f"{(z_mp_cpu_col - z_np).mean()=}")

    with Timer('mumpy.matmul_cuda'):
        z_mp_gpu = mp.matmul_cuda(x, y)
    print(f"{(z_mp_gpu - z_np).mean()=}")


if __name__ == '__main__':
    main()
