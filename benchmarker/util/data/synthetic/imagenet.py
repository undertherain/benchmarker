"""Generate synthetic data for ImageNet type problems"""

import numpy as np


def get_data(params):
    """Generate synthetic data for ImageNet type problems. Import this
    function in the `data.py` of the problem, so it can be called by
    `INeuralNet`.

    """

    if isinstance(params["problem"]["size"], int):
        if params["channels_first"]:
            params["problem"]["size"] = (params["problem"]["size"], 3, 224, 224)
        else:
            params["problem"]["size"] = (params["problem"]["size"], 224, 224, 3)

    shape = (params["problem"]["cnt_batches_per_epoch"], params["batch_size"])
    shape = shape + params["problem"]["size"][1:]

    X = np.random.random(shape).astype(np.float32)
    Y = np.random.random(shape[:2]).astype(np.int64)
    return X, Y
