import torch.nn as nn
import torch.nn.functional as F


class Net4Inference(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def __call__(self, x):
        return F.softmax(self.net(x))


class Net4Train(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.criterion = nn.CrossEntropyLoss()

    def __call__(self, x, t):
        output = self.net(x)
        return self.criterion(output, t)
