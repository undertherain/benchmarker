from benchmarker.modules.problems.images_randomized import gen_images


def get_data(params):
    assert params["problem"]["size"][1] == 3
    assert params["problem"]["size"][2] == 224
    assert params["problem"]["size"][3] == 224
    return gen_images(params)
    """generates sinthetic dataset"""

    # if "size" in params["problem"]:
    #     cnt_samples = params["problem"]["size"]
    # else:
    #     cnt_samples = 1024

    # return get_cubes(dims=2, edge=224, channels=3,
    #                  cnt_samples=cnt_samples,
    #                  channels_first=params["channels_first"], onehot=False)
