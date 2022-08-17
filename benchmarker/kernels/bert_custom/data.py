import numpy as np


def get_data(params):
    """generates sinthetic dataset"""
    # input size -> cnt_sequences, len_suqence, cnt_dimentsions
    # transform into (seq_len, batch) x cnt_batches
    if isinstance(params["problem"]["size"], int):
        params["problem"]["size"] = (params["problem"]["size"], 128)
    assert params["problem"]["size"][0] % params["batch_size"] == 0
    params["problem"]["len_sequence"] = params["problem"]["size"][1]
    cnt_batches = params["problem"]["size"][0] // params["batch_size"]
    shape = (params["batch_size"],
             params["problem"]["len_sequence"],
             )
    X = np.random.random(shape).astype(np.int64)
    res = [{"input_ids": X, "labels": X} for i in range(cnt_batches)]
    return res





    # assert params["problem"]["size"][0] % params["batch_size"] == 0
    # params["problem"]["len_sequence"] = params["problem"]["size"][1]
    # cnt_batches = params["problem"]["size"][0] // params["batch_size"]
    # shape = (params["problem"]["len_sequence"],
    # data = [
    #     {"iu"}
    # X = [np.random.randint(low=0, high=params["problem"]["cnt_tokens"], size=shape) for i in range(cnt_batches)] 
    # Y = [np.ones((params["problem"]["len_sequence"], params["batch_size"])) for i in range(cnt_batches)]
    # return X, Y
