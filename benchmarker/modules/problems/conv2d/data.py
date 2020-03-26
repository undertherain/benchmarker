import numpy as np
from benchmarker.util.data.cubes import get_cubes

# TODO: reuse single code for Imagenet-like data


def get_data(params):
    """generates sinthetic dataset"""
    # (cnt_batches, batch, channels, x, h)
    assert len(params["problem"]["size"]) == 4

    shape = (params["problem"]["cnt_batches_per_epoch"],
             params["batch_size"],
             params["problem"]["size"][1],
             params["problem"]["size"][2],
             params["problem"]["size"][3])
    X = np.random.random(shape).astype(np.float32)
    Y = np.random.random((params["problem"]["size"][0])).astype(np.int64)
    return X, Y

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
