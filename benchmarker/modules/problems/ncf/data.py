import numpy as np


def get_data(params):
    """generates synthetic dataset"""
    #input size -> nb_users, nb_items, factors, layers
    # transform into (nb_users, nb_items, factors, layers) x cnt_batches
    assert params["problem"]["size"][0] % params["batch_size"] == 0
    params["problem"]["nb_users"] = params["problem"]["size"][1]
    params["problem"]["nb_items"] = params["problem"]["size"][1]
    cnt_batches = params["problem"]["size"][0] // params["batch_size"]
    #shape = (cnt_bathces, 2,
             params["problem"]["nb_users"],
             params["problem"]["nb_items"])
    #

    mp.array([ (randint(0..cnt_users), randint(0..cnt_items), rand(0_max_score)) ]  for _ in range size_data])
    resahpe(cnt_batches, batch_size, -1)
    data  = one_item x batch_size x cnt_batches 
    #X = [np.array([np.random.random(shape).astype(np.int64), np.random.random(shape).astype(np.int64)]) for _ in range(cnt_batches)]
    Y = np.ones((cnt_batches,params["batch_size"]))
    return X, Y
