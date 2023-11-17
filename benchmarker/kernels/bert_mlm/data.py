import numpy as np


def get_data(params):
    params["problem"]["len_sequence"] = params["problem"]["sample_shape"]
    cnt_batches = params["problem"]["cnt_batches_per_epoch"]
    shape = (params["batch_size"],
             params["problem"]["len_sequence"],
             )
    X = np.random.random(shape).astype(np.int64)
    res = [{"input_ids": X, "labels": X} for i in range(cnt_batches)]
    return res
