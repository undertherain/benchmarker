from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseWrapper(nn.Module):
    def __init__(self, mode, net, criterion):
        super().__init__()
        self.mode = mode
        self.net = net
        self.criterion = criterion

    def __call__(self, **kwargs):
        if self.mode == "inference":
            return self.call_inference(**kwargs)
        else:
            return self.call_training(**kwargs)

    def call_inference(self, x, labels):
        outs = self.net(x)
        if isinstance(outs, OrderedDict):
            outs = outs["out"]
        return outs

    def call_training(self, x, labels):
        if isinstance(x, dict):
            outs = self.net(**x)
        else:
            outs = self.net(x)
        # TODO(alex): figure this out. there's a reason why backward
        # finction is returned precompiled? is it correct to ignore
        # it?
        if isinstance(outs, OrderedDict):
            outs = outs["out"]
        if outs.layout == torch._mkldnn:
            outs = outs.to_dense()
        loss = self.criterion(outs, labels)
        return loss


class Classifier(BaseWrapper):
    def __init__(self, mode, net):
        super().__init__(mode, net, nn.CrossEntropyLoss())

    def call_infererance(self, x, labels):
        outs = self.net(x)
        if isinstance(outs, OrderedDict):
            outs = outs["out"]
        return F.softmax(outs, dim=-1)


class Regression(BaseWrapper):
    def __init__(self, mode, net):
        super().__init__(mode, net, loss=nn.MSELoss())


class Recommender(BaseWrapper):
    def __init__(self, mode, net):
        super().__init__(mode, net, nn.BCEWithLogitsLoss())

    def call_inference(self, x, labels):
        outs = self.net(x)
        if isinstance(outs, OrderedDict):
            outs = outs["out"]
        return torch.sigmoid(outs)
