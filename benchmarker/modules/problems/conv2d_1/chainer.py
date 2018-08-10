import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from timeit import default_timer as timer

class Net(chainer.Chain):
    def __init__(self, train=True):
        super(Net, self).__init__(
            conv1=L.Convolution2D(1, 32, 2), #Convolution2D(in_channels, out_channels, ksize, stride=1, pad=0, wscale=1, bias=0, nobias=False, use_cudnn=True, initialW=None, initial_bias=None, deterministic=False)
            conv2=L.Convolution2D(None, 32, 2),
            l=L.Linear(None, 2),
        )
        self.train = train

    def __call__(self, x):
        h = x
        h = self.conv1(h)
        h = F.relu(h)
        #h = F.max_pooling_2d(h,2)
        h = self.conv2(h)
        h = F.relu(h)
        h = self.l(h)
        return h
