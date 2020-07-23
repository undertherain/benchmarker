import numpy as np


def get_data(params):
    cnt_batches = params["problem"]["cnt_batches_per_epoch"]
    batch_size = params["batch_size"]

    shape = params["input_shape"]
    if params["channels_first"]:
        shape = shape[-1:] + shape[:-1]
        # params["input_shape"] = shape # should this be updated?

    shape = (cnt_batches, batch_size) + shape
    X = np.random.random(shape).astype(np.float32)

    shape = (cnt_batches, batch_size, params["target_size"])
    Y = np.random.random(shape).astype(np.float32)

    return X, Y
