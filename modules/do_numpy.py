import numpy as np
from timeit import default_timer as timer


def run(params):
    if params["problem"] != "2048": # todo change to actual parameter
        raise Exception("only 2048 problem is defined for numpy")
    a = np.random.random((2048, 2048))
    b = np.random.random((2048, 2048))
    nb_epoch = 1

    start = timer()
    for i in range(nb_epoch):
        c = a @ b
    end = timer()

    params["time"] = (end-start)/nb_epoch
    return params
