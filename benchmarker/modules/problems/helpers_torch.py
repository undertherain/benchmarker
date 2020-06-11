import torch.nn as nn


class Net4Inference(nn.Module):
    def __init__(self, net, head=id):
        super().__init__()
        self.net = net
        self.head = head

    def __call__(self, x):
        return self.head(self.net(x))


class Net4Train(nn.Module):
    def __init__(self, net, criterion=nn.CrossEntropyLoss()):
        super().__init__()
        self.net = net
        self.criterion = criterion

    def __call__(self, x, t):
        output = self.net(x)
        return self.criterion(output, t)
