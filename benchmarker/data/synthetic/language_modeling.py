import numpy as np


def get_data(params):
    assert params["problem"]["size"][0] % params["batch_size"] == 0
    if isinstance(params["problem"]["size"], int):
        params["problem"]["size"] = (params["problem"]["size"], 128)
    # TODO: return ints in cnt_labels range
    params["problem"]["len_sequence"] = params["problem"]["size"][1]
    cnt_batches = params["problem"]["size"][0] // params["batch_size"]
    shape = (params["batch_size"],
             params["problem"]["len_sequence"],
             )
    # TODO: use max token id here
    X = np.random.randint(low=0, high=1024, size=shape).astype(np.int64)
    res = [{"input_ids": X, "labels": X} for i in range(cnt_batches)]
    return res
