import sys
import begin

sys.path.append("../../../data_helpers/")
from cubes import get_cubes


def get_data():
    return get_cubes(dims=2, edge=128, channels=1, cnt=2048)


@begin.start
def main():
    data = get_data()
    print(data[0].shape)
