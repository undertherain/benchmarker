import numpy as np


def get_data(params):
    """generates synthetic dataset"""
    #input size -> nb_users, nb_items, factors, layers
    # transform into (nb_users, nb_items, factors, layers) x cnt_batches
    assert params["problem"]["size"][0] % params["batch_size"] == 0
    params["problem"]["nb_users"] = params["problem"]["size"][1]
    params["problem"]["nb_items"] = params["problem"]["size"][2]
    cnt_batches = params["problem"]["size"][0] // params["batch_size"]
    shape = (cnt_batches,
             params["problem"]["nb_users"],
             params["problem"]["nb_items"])
    X = np.random.random(shape).astype(np.int64)
    Y = np.ones((cnt_batches,params["batch_size"]))
    return X, Y
