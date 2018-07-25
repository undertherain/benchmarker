from benchmarker.util.data.cubes import get_cubes

#def get_data(params):
#    X, Y = get_cubes(dims=2, edge=224, channels=3, cnt=10*1024, channels_first=params["channels_first"])
#    Y = to_one_hot(Y, cnt_classes=1000)
#    return X, Y

def get_data(params):
    return get_cubes(dims=2, edge=224, channels=3, cnt_samples=1024, channels_first=params["channels_first"], onehot=False)
