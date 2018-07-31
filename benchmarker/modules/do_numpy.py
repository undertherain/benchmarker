# -*- coding: utf-8 -*-
"""NumPy support.
"""

from timeit import default_timer as timer
import numpy as np


def run(params):
    if params["problem"]["name"] != "2048": # todo change to actual parameter
        raise Exception("only 2048 problem is defined for numpy")
    if "nb_gpus" in params:
        if params["nb_gpus"] > 0:
            raise Exception("Numpy framework does not work with GPU back-end")

    size = 2048

    matrix_in_1 = np.random.random((size, size))
    matrix_in_2 = np.random.random((size, size))

    nb_epoch = 1

    start = timer()
    for _ in range(nb_epoch):
        matrix_out = matrix_in_1 @ matrix_in_2
    end = timer()

    params["time"] = (end - start) / nb_epoch
    return params
