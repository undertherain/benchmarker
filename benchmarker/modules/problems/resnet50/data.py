from benchmarker.modules.problems.images_randomized import gen_images


def get_data(params):
    if params["channels_first"]:
        if isinstance(params["problem"]["size"], int):
            params["problem"]["size"] = (params["problem"]["size"], 3, 224, 224)
        assert params["problem"]["size"][1] == 3
        assert params["problem"]["size"][2] == 224
        assert params["problem"]["size"][3] == 224
    else:
        if isinstance(params["problem"]["size"], int):
            params["problem"]["size"] = (params["problem"]["size"], 224, 224, 3)
        assert params["problem"]["size"][1] == 224
        assert params["problem"]["size"][2] == 224
        assert params["problem"]["size"][3] == 3
    return gen_images(params)
    """generates sinthetic dataset"""

    # if "size" in params["problem"]:
    #     cnt_samples = params["problem"]["size"]
    # else:
    #     cnt_samples = 1024

    # return get_cubes(dims=2, edge=224, channels=3,
    #                  cnt_samples=cnt_samples,
    #                  channels_first=params["channels_first"], onehot=False)
