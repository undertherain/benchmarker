import numpy as np


def get_data(params):
    """generates synthetic dataset"""
    # input size -> nb_users, nb_items, factors, layers
    # transform into (nb_users, nb_items, factors, layers) x cnt_batches
    nb_users = params["problem"]["nb_users"]
    nb_items = params["problem"]["nb_items"]
    batch_size = params["batch_size"]
    assert params["problem"]["size"] % batch_size == 0
    cnt_batches = params["problem"]["size"] // batch_size
    # shape = (cnt_bathces, 2,

    X = []
    Y = []
    for _ in range(cnt_batches):
        users = np.random.randint(0, nb_users, size=batch_size, dtype=np.int64)
        items = np.random.randint(0, nb_items, size=batch_size, dtype=np.int64)
        batch_x = np.array([users, items])
        batch_y = np.random.random(batch_size)
        X.append(batch_x)
        Y.append(batch_y)

    X = np.array(X)
    Y = np.array(Y)
    print("param[problems]", params["problem"])
    print("X.shape", X.shape)
    print("Y.shape", Y.shape)
    # mp.array([ (randint(0..cnt_users), randint(0..cnt_items), rand(0_max_score)) ]  for _ in range size_data])
    # resahpe(cnt_batches, batch_size, -1)
    # data  = one_item x batch_size x cnt_batches
    # X = [np.array([np.random.random(shape).astype(np.int64), np.random.random(shape).astype(np.int64)]) for _ in range(cnt_batches)]
    # Y = np.ones((cnt_batches, params["batch_size"]))
    return X, Y
