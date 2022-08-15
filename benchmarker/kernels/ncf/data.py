import numpy as np


def get_data(params):
    """generates synthetic dataset"""
    nb_users = params["problem"]["nb_users"]
    nb_items = params["problem"]["nb_items"]
    batch_size = params["batch_size"]
    cnt_batches = params["problem"]["size"] // batch_size
    assert params["problem"]["size"] % batch_size == 0

    data = []
    for _ in range(cnt_batches):
        users = np.random.randint(0, nb_users, size=batch_size, dtype=np.int64)
        items = np.random.randint(0, nb_items, size=batch_size, dtype=np.int64)
        batch_x = np.array([users, items])
        batch_y = np.random.random(batch_size).reshape([-1, 1])
        data.append({"x": batch_x, "labels": batch_y})
    return data
