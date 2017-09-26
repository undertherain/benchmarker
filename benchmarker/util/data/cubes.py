import numpy as np


def get_cubes(dims=3, edge=8, channels=1, cnt=16, cnt_classes=2, channels_first=True, onehot=False):
    if channels_first:
        shape = tuple([cnt, channels] + [edge] * dims)
    else:
        shape = tuple([cnt] + [edge] * dims + [channels])
    X = np.zeros(shape, dtype=np.float32)
    Y = np.zeros(cnt, dtype=np.int32)
    for i in range(cnt // 2):
        X[i*2, :] = np.ones(shape[1:])
        Y[i % cnt_classes] = True
    return X, Y


def to_one_hot(Y, cnt_classes):
    Y_new = np.zeros((Y.shape[0], cnt_classes))
    Y_new[np.arange(Y.shape[0]), Y] = 1
    return Y_new


def main():
    x, y = get_cubes(2, 1, 4)
    print(x.shape)


if __name__ == "__main__":
    main()
