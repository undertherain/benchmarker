"""sinthetic dataset for vgg16"""

from benchmarker.util.data.cubes import get_cubes


def get_data(params):
    """generates sinthetic dataset"""
    if "size" in params["problem"]:
        cnt_samples = params["problem"]["size"]
    else:
        cnt_samples = 1024
    return get_cubes(dims=2, edge=300, channels=3, cnt_samples=cnt_samples,
                     channels_first=params["channels_first"], onehot=False)
