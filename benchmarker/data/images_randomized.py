import numpy as np


# TODO: this repeats code from .synthetic
def gen_data(params):
    """generates sinthetic dataset"""
    # input size -> cnt_sequences, len_sequence, cnt_dimensions
    # transform into (seq_len, batch, cnt_dimentions) x cnt_batches
    assert params["problem"]["size"][0] % params["batch_size"] == 0
    params["problem"]["len_sequence"] = params["problem"]["size"][1]
    cnt_batches = params["problem"]["size"][0] // params["batch_size"]
    shape = (
        params["problem"]["len_sequence"],
        params["batch_size"],
        params["problem"]["size"][2],
    )
    data = [{"x": np.random.random(shape).astype(np.float32),
             "labels": np.ones((params["batch_size"]))} for i in range(cnt_batches)]
    return data
