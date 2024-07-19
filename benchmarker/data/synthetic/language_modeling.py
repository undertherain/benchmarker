import numpy as np


def get_data(params):
    # if isinstance(params["problem"]["size"], int):
    #     params["problem"]["size"] = (params["problem"]["size"], 128)
    # TODO: return ints in cnt_labels range
    assert len(params["problem"]["sample_shape"]) == 1
    params["problem"]["len_sequence"] = params["problem"]["sample_shape"][0]
    cnt_batches = params["problem"]["cnt_samples"] // params["batch_size"]
    shape = (params["batch_size"],
             params["problem"]["len_sequence"],
             )
    # TODO: use max token id here
    X = np.random.randint(low=0, high=1024, size=shape).astype(np.int64)
    res = [{"input_ids": X, "labels": X} for i in range(cnt_batches)]
    return res
