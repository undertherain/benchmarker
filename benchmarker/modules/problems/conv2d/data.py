#import numpy as np
# from benchmarker.util.data.cubes import get_cubes

# TODO: reuse single code for Imagenet-like data
from benchmarker.modules.problems.images_randomized import gen_images


def get_data(params):
    # TODO: expand to tuple if needed
    return gen_images(params)



    # TODO: restore making images of different patterns to correspong to different labels
    # if isinstance(params["problem"]["size"], int):
    #     return get_cubes(dims=2, edge=224, channels=3,
    #                      cnt_samples=params["problem"]["size"],
    #                      channels_first=params["channels_first"], onehot=False)
    # else:
    #     return get_cubes(dims=2, edge=params["problem"]["size"][2],
    #                      channels=params["problem"]["size"][1],
    #                      cnt_samples=params["problem"]["size"][0],
    #                      channels_first=params["channels_first"], onehot=False)
