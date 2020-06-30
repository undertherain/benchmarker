import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class Net4Inference(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def __call__(self, x):
        return self.net(x)


class Net4Train(nn.Module):
    def __init__(self, net, criterion):
        super().__init__()
        self.net = net
        self.criterion = criterion

    def __call__(self, x, t):
        outs = self.net(x)
        # TODO: figure this out. there's a reason why backward finction is returned
        # precompiled? is it correct to ignore it?
        if isinstance(outs, OrderedDict):
            outs = outs["out"]
        loss = self.criterion(outs, t)
        return loss


def Net4Both(params, net, inference, training):
    if params["mode"] == "inference":
        return inference(net)
    else:
        return training(net)


class ClassifierInference(Net4Inference):
    def __call__(self, x):
        outs = self.net(x)
        # TODO: figure this out. there's a reason why backward finction is returned
        # precompiled? is it correct to ignore it?
        if isinstance(outs, OrderedDict):
            outs = outs["out"]
        return F.softmax(outs, dim=-1)


class ClassifierTraining(Net4Train):
    def __init__(self, net):
        super().__init__(net, nn.CrossEntropyLoss())


def Classifier(params, net):
    """Returns an inference or training classifier."""
    return Net4Both(params, net, ClassifierInference, ClassifierTraining)
