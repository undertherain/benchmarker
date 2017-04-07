import chainer
import chainer.functions as F
import chainer.links as L


class Net(chainer.Chain):

    def __init__(self, train=True):
        super(Net, self).__init__(
            conv1=L.Convolution2D(1, 64, 2),
            conv2=L.Convolution2D(None, 64, 2),
            conv3=L.Convolution2D(None, 64, 2),
            conv4=L.Convolution2D(None, 64, 2),
            l=L.Linear(None, 1),
        )
        self.train = train

    def __call__(self, x):
        h = x
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 2)
        h = self.conv3(h)
        h = F.relu(h)
        h = self.conv4(h)
        h = F.relu(h)
        h = self.l(h)
        return h
