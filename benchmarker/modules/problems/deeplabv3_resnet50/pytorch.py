import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

from ..helpers_torch import Classifier


def get_kernel(params, unparsed_args):
    # TODO: assert unparsed args are empty
    # TODO: make in a per-pixel classifier for training
    # TODO: cnt classes as parameter
    params["problem"]["cnt_classes"] = 21
    net = deeplabv3_resnet50(num_classes=params["problem"]["cnt_classes"])
    return Classifier(params, net)
