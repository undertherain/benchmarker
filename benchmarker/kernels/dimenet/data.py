import numpy as np

cnt_atoms = 10


def get_molecule():
    atoms = np.random.randint(low=1, high=60, size=cnt_atoms, dtype=np.int64)
    positions = np.random.random((cnt_atoms, 3)).astype(np.float32)
    return {"z": atoms, "pos": positions}


def get_data(params):
    cnt_batches = 4
    X = [get_molecule() for i in range(cnt_batches)]
    Y = [np.ones((params["batch_size"])).astype(np.float32) / 32 for i in range(cnt_batches)]
    return X, Y
