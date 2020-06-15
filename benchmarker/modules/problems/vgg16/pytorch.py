import torch
import torch.nn.functional as F
import torchvision.models as models

from ..helpers_torch import Classifier


def get_kernel(params, unparsed_args):
    net = models.vgg16()
    return Classifier(params, net)
