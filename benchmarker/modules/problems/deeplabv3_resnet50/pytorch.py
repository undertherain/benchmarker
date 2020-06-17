from torchvision.models.segmentation import deeplabv3_resnet50


def get_kernel(params, unparsed_args):
    net = deeplabv3_resnet50()
    # TODO: assert unparsed args are empty
    # TODO: make in a per-pixel classifier for training
    assert params["mode"] == "inference"
    return net
