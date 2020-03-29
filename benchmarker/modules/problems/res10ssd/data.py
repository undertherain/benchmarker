"""sinthetic dataset for vgg16"""

#from benchmarker.util.data.cubes import get_cubes
from benchmarker.modules.problems.images_randomized import gen_images


def get_data(params):
    if params["channels_first"]:
        if isinstance(params["problem"]["size"], int):
            params["problem"]["size"] = (params["problem"]["size"], 3, 300, 300)
        assert params["problem"]["size"][1] == 3
    else:
        if isinstance(params["problem"]["size"], int):
            params["problem"]["size"] = (params["problem"]["size"], 300, 300, 3)
        assert params["problem"]["size"][3] == 3        
    return gen_images(params)    # """generates sinthetic dataset"""
    # if "size" in params["problem"]:
    #     cnt_samples = params["problem"]["size"]
    # else:
    #     cnt_samples = 1024
    # return get_cubes(dims=2, edge=300, channels=3, cnt_samples=cnt_samples,
    #                  channels_first=params["channels_first"], onehot=False)
