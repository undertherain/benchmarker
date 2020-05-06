import numpy as np


def get_cubes(dims=3, edge=8, channels=1, cnt_samples=16, cnt_classes=2, channels_first=True, onehot=False):
    if channels_first:
        shape = tuple([cnt_samples, channels] + [edge] * dims)
    else:
        shape = tuple([cnt_samples] + [edge] * dims + [channels])
    X = np.zeros(shape, dtype=np.float32)
    Y = np.zeros(cnt_samples, dtype=np.int32)
    for i in range(cnt_samples):
        if i % 2 == 1:
            X[i, :] = np.ones(shape[1:])
            Y[i] = 1
    return X, Y


def to_one_hot(Y, cnt_classes):
    Y_new = np.zeros((Y.shape[0], cnt_classes))
    Y_new[np.arange(Y.shape[0]), Y] = 1
    return Y_new


def test():
    params = {
        "dims": 2,
        "edge": 1,
        "channels": 4,
    }
    X, Y = get_cubes(**params)
    assert X.shape == (16, 4, 1, 1)
    assert Y.shape == (16,)


if __name__ == "__main__":
    test()
