import begin
from benchmarker.util.data.cubes import get_cubes


def get_data(params):
    return get_cubes(dims=2, edge=128, channels=1, cnt=10*1024, channels_first=params["channels_first"])


@begin.start
def main():
    data = get_data()
    print(data[0].shape)
