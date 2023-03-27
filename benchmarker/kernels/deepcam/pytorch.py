from ..helpers_torch import Classifier
from .deeplab_xception import DeepLabv3_plus


def get_kernel(params):
    # TODO: cnt classes as parameter
    params["problem"]["cnt_classes"] = 21
    net = DeepLabv3_plus(n_classes=params["problem"]["cnt_classes"])
    return Classifier(params["mode"], net)
