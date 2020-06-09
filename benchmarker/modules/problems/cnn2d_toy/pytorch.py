import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2)
        # TODO: make sure we check cnt_classes
        self.dense1 = nn.Linear(1577088, 2)

    def __call__(self, x):
        h = x
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        h = F.relu(h)
        h = torch.flatten(h, 1)
        h = self.dense1(h)
        return h


class Net4Inference(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Net()

    def __call__(self, x):
        return F.softmax(self.net(x))


class Net4Train(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Net()
        self.criterion = nn.CrossEntropyLoss()

    def __call__(self, x, t):
        output = self.net(x)
        return self.criterion(output, t)


def get_kernel(params, unparsed_args=None):
    if params["mode"] == "inference":
        net = Net4Inference()
    else:
        net = Net4Train()
    return net
