import numpy as np


def get_data(params):
    """generates sinthetic dataset"""
    # (seq_len, batch, input_size)
    # pre-batch elements
    assert len(params["problem"]["size"]) == 2
    params["problem"]["len_sequence"] = 1

    shape = (params["problem"]["size"][0],
             params["problem"]["len_sequence"],
             params["batch_size"],
             params["problem"]["size"][1])
    X = np.random.random(shape).astype(np.float32)
    Y = np.random.random((params["problem"]["size"][0])).astype(np.int64)
    return X, Y
