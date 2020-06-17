import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

from ..helpers_torch import Classifier


def get_kernel(params, unparsed_args):
    assert unparsed_args == []
    # TODO: cnt classes as parameter
    params["problem"]["cnt_classes"] = 21
    net = deeplabv3_resnet50(num_classes=params["problem"]["cnt_classes"])
    return Classifier(params, net)
