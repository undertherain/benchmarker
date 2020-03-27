import numpy as np
from benchmarker.util.data.cubes import get_cubes


# TODO: get back generation of somewhat patterned images
def gen_images(params):
    """generates sinthetic dataset"""
    # (cnt_batches, batch, channels, x, h)
    assert len(params["problem"]["size"]) == 4

    shape = (params["problem"]["cnt_batches_per_epoch"],
             params["batch_size"],
             params["problem"]["size"][1],
             params["problem"]["size"][2],
             params["problem"]["size"][3])
    X = np.random.random(shape).astype(np.float32)
    Y = np.random.random(shape[:2]).astype(np.int64)
    return X, Y