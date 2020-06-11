import torch
import torch.nn.functional as F
import torchvision.models as models

from ..helpers_torch import Net4Both


def get_kernel(params, unparsed_args):
    return Net4Both(
        params,
        models.resnet50(),
        lambda t: F.softmax(t, dim=-1),
        torch.nn.CrossEntropyLoss(),
    )
