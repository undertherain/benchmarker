import numpy as np


def get_data(params):
    """generates sinthetic dataset"""
    #input size -> cnt_sequences, len_suqence, cnt_dimentsions
    # transform into (seq_len, batch) x cnt_batches
    assert params["problem"]["size"][0] % params["batch_size"] == 0
    params["problem"]["len_sequence"] = params["problem"]["size"][1]
    cnt_batches = params["problem"]["size"][0] // params["batch_size"]
    shape = (cnt_batches,
             params["problem"]["len_sequence"],
             params["batch_size"])
    X = np.random.random(shape).astype(np.int64)
    Y = np.ones((cnt_batches, params["batch_size"]))
    return X, Y
