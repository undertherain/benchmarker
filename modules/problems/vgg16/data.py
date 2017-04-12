import sys
import begin
sys.path.append("../../../data_helpers")
from cubes import get_cubes, to_one_hot


def get_data():
    X, Y = get_cubes(dims=2, edge=224, channels=3, cnt=512, data_format = "channels_last")
    Y = to_one_hot(Y,cnt_classes=1000)
    return X, Y


@begin.start
def main():
    data = get_data()
    print (data[0].shape)

