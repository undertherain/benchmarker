import numpy as np

max_atom_index = 95


def get_batch(batch_size, cnt_atoms_in_sample):
    cnt_atoms_in_batch = cnt_atoms_in_sample * batch_size
    atoms = np.random.randint(
        low=1, high=max_atom_index, size=cnt_atoms_in_batch, dtype=np.int64
    )
    positions = (np.random.random((cnt_atoms_in_batch, 3)).astype(np.float32) - 0.5) * 4
    batch = (
        np.ones((batch_size, cnt_atoms_in_sample), np.int64)
        * np.arange(batch_size)[:, np.newaxis]
    )
    batch = batch.ravel()
    return {"z": atoms, "pos": positions, "batch": batch}


def get_data(params):
    # This should be number item in problem size
    cnt_batches = params["problem"]["cnt_batches_per_epoch"]
    X = [
        get_batch(params["batch_size"], params["problem"]["cnt_atoms_in_sample"])
        for i in range(cnt_batches)
    ]
    Y = [
        np.random.random((params["batch_size"])).astype(np.float32) - 0.5
        for i in range(cnt_batches)
    ]
    return X, Y
