import torchvision.models as models

from ..helpers_torch import Classifier


def get_kernel(params):
    net = models.resnet50()
    return Classifier(params["mode"], net)
