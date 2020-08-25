import torchvision.models as models

from ..helpers_torch import Classifier


def get_kernel(params, unparsed_args):
    # TODO(Alex): think of a way to do it in a reusable fashion
    assert not unparsed_args
    net = models.resnet50()
    return Classifier(params, net)
