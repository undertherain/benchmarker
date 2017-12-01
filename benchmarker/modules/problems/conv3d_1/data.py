from benchmarker.util.data.cubes import get_cubes


def get_data(params):
    return get_cubes(dims=3, edge=64, channels=1, cnt=10*1024)
