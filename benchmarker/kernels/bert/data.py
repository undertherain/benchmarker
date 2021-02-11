import numpy as np


def get_data(params):
    if isinstance(params["problem"]["size"], int):
        params["problem"]["size"] = (params["problem"]["size"], 128)
    assert params["problem"]["size"][0] % params["batch_size"] == 0
    params["problem"]["len_sequence"] = params["problem"]["size"][1]
    cnt_batches = params["problem"]["size"][0] // params["batch_size"]
    shape = (cnt_batches,
             params["batch_size"],
             params["problem"]["len_sequence"],
             )
    X = np.random.random(shape).astype(np.int64)
    Y = np.ones((cnt_batches, params["batch_size"]), dtype=np.int64)
    return X, Y
