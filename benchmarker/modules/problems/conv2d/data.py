from benchmarker.util.data.cubes import get_cubes

# TODO: reuse single code for Imagenet-like data


def get_data(params):
    """generates sinthetic dataset"""

    if isinstance(params["problem"]["size"], int):
        return get_cubes(dims=2, edge=224, channels=3,
                         cnt_samples=params["problem"]["size"],
                         channels_first=params["channels_first"], onehot=False)
    else:
        return get_cubes(dims=2, edge=params["problem"]["size"][2],
                         channels=params["problem"]["size"][1],
                         cnt_samples=params["problem"]["size"][0],
                         channels_first=params["channels_first"], onehot=False)
