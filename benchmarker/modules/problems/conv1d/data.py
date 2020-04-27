#import numpy as np
# from benchmarker.util.data.cubes import get_cubes

# TODO: reuse single code for Imagenet-like data
from benchmarker.modules.problems.images_randomized import gen_images


def get_data(params):
    return gen_images(params)
