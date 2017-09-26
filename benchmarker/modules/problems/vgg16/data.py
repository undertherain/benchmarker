import sys
import begin
sys.path.append("../../../data_helpers")
from cubes import get_cubes, to_one_hot


def get_data(params):
    X, Y = get_cubes(dims=2, edge=224, channels=3, cnt=10*1024, channels_first=params["channels_first"])
    Y = to_one_hot(Y, cnt_classes=1000)
    return X, Y


@begin.start
def main():
    data = get_data()
    print(data[0].shape)
