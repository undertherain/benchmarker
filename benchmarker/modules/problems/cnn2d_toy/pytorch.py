import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, train=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2)
        self.dense1 = nn.Linear(222, 2)

    def __call__(self, x):
        h = x
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        h = F.relu(h)
        h = self.dense1(h)
        return h


def get_kernel(params, unparsed_args=None):
    net = Net()
    # TODO: add 
    return net
