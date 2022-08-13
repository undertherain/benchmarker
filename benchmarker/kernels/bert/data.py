import numpy as np


def get_data(params):
    if isinstance(params["problem"]["size"], int):
        params["problem"]["size"] = (params["problem"]["size"], 128)
    assert params["problem"]["size"][0] % params["batch_size"] == 0
    params["problem"]["len_sequence"] = params["problem"]["size"][1]
    cnt_batches = params["problem"]["size"][0] // params["batch_size"]
    shape = (params["batch_size"],
             params["problem"]["len_sequence"],
             )
    # TODO: this should be within vocab size
    X = np.random.random(shape).astype(np.int64)
    # TODO: return ints in cnt_labels range
    Y = np.ones(params["batch_size"], dtype=np.int64)
    res = [{"input_ids": X, "labels": Y} for i in range(cnt_batches)]
    return res
