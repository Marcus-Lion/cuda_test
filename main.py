import logging

import torch
from numba import jit, cuda
import numpy as np
# to measure exec time
from timeit import default_timer as timer

n = 50000000


# normal function to run on cpu
def func(a):
    for i in range(n):
        a[i] += 1

    # function optimized to run on gpu


@jit()
def func2(a):
    for i in range(n):
        a[i] += 1


if __name__ == "__main__":
    gpu = False
    if torch.cuda.is_available():
        logging.info("CUDA is available!")
        gpu = True
    else:
        logging.warning("CUDA is not available.")


    a = np.ones(n, dtype=np.float64)

    start = timer()
    func(a)
    print("without GPU:", timer() - start)

    dev = cuda.select_device(0)
    print(dev)

    start = timer()
    func2(a)
    print("with GPU:", timer() - start)