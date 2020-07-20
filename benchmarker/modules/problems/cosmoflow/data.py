import numpy as np


def get_data(params):
    shape = (params["problem"]["cnt_batches_per_epoch"], params["batch_size"])
    shape += params["input_shape"]
    X = np.random.random(shape).astype(np.float32)

    shape = (
        params["problem"]["cnt_batches_per_epoch"],
        params["batch_size"],
        params["target_size"],
    )
    Y = np.random.random(shape).astype(np.float32)

    return X, Y
