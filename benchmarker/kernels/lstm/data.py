import numpy as np

# TODO: move this to data
from benchmarker.kernels.images_randomized import gen_data


def get_data(params):
    return gen_data(params)
