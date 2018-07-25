from benchmarker.util.data.cubes import get_cubes

def get_data(params):
    return get_cubes(dims=2, edge=224, channels=3, cnt_samples=1024, channels_first=params["channels_first"], onehot=False)
