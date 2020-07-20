### !!! STUB !!! ###
import numpy as np


def get_data(params):
    shape = (1, 2)
    num_cls = 3
    X = np.random.random(shape).astype(np.float32)
    Y = np.random.randint(0, num_cls, shape[:2])
    return X, Y
