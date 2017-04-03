import numpy as np
import chainer
import chainer
import chainer.functions as F
import chainer.links as L

class Net(chainer.Chain):
    def __init__(self, train=True):
        super(Net, self).__init__(
            conv1=L.Convolution2D(1, 64, 2), #Convolution2D(in_channels, out_channels, ksize, stride=1, pad=0, wscale=1, bias=0, nobias=False, use_cudnn=True, initialW=None, initial_bias=None, deterministic=False)
            conv2=L.Convolution2D(None, 64, 2),
            conv3=L.Convolution2D(None, 64, 2),
            conv4=L.Convolution2D(None, 64, 2),
            #l1=L.Linear(None, 128),   #this is ugly
            l=L.Linear(None, 1),   #this is ugly
        )
        self.train = train
    def __call__(self, x):
        h = x
        
        h = self.conv1(h)
        h = F.relu(h)
        #h = F.max_pooling_2d(h,2)
        h = self.conv2(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 2)
        h = self.conv3(h)
        h = F.relu(h)
        #h = F.max_pooling_2d(h,2)
        h = self.conv4(h)
        h = F.relu(h)
      
      #h = F.flatten(h)
#        h = F.relu(self.conv3(h))
 #       h = F.droputog cjainer flatte(F.max_pooling_2d(F.relu(self.conv4(h)), 2),ratio=0.3)
        #h = F.relu(self.l1(h))
        h = self.l(h)
        return h
