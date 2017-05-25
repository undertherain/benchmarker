import chainer
import chainer.functions as F
import chainer.links as L

class Net(chainer.Chain):
    def __init__(self, train=True):
        super(Net, self).__init__(
            conv1=L.ConvolutionND(3, 1, 16, 3), #Convolution2D(in_channels, out_channels, ksize, stride=1, pad=0, wscale=1, bias=0, nobias=False, use_cudnn=True, initialW=None, initial_bias=None, deterministic=False)
            conv2=L.ConvolutionND(3, 16, 16, 3),
            #l1=L.Linear(None, 128),   #this is ugly
            l=L.Linear(None, 1),   #this is ugly
        )
        self.train = train
    def __call__(self, x):
        h = F.relu(self.conv1(x))
        #h = F.max_pooling_2d(h,2)
        h = F.relu(self.conv2(h))
#        h = F.relu(self.conv3(h))
#        h = F.dropout(F.max_pooling_2d(F.relu(self.conv4(h)), 2),ratio=0.3)
        #h = F.relu(self.l1(h))
        h = self.l(h)
        return h
