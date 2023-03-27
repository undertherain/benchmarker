import numpy as np


def get_data(params):
    cnt_batches = params["problem"]["cnt_batches_per_epoch"]
    batch_size = params["batch_size"]

    shape = params["input_shape"]
#    if params["channels_first"]:
#        shape = shape[-1:] + shape[:-1]
        # params["input_shape"] = shape # should this be updated?

    shape = (batch_size,) + shape
    data = [{"x": np.random.random(shape).astype(np.float32),
             "labels": np.random.random((batch_size, params["target_size"])).astype(np.float32)}
            for i in range(cnt_batches)]
    return data
