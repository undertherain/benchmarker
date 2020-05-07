from benchmarker.modules.problems.images_randomized import gen_images

# TODO(vatai): I copied this from
# benchmarker/modules/problems/resnet50/data.py so this is duplicated
# code which should be refactored (see github issue #18).


def get_data(params):
    """Generates synthetic dataset."""
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
