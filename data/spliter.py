import torch as th
import numpy as np

def index_to_mask(index, size):
    if th.is_tensor(index):
        mask = th.zeros(size, dtype=th.int)
        mask[index] = 1
    else:
        mask = np.zeros(size)
        mask[index] = 1
    return mask