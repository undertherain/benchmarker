# -*- coding: utf-8 -*-
"""MxNet support.

This module integrates MxNet framework into the benchmarker
"""

import importlib
from timeit import default_timer as timer
import mxnet as mx
from .i_neural_net import INeuralNet


class DoMxnet(INeuralNet):

    def __init__(self, params):
        super().__init__(params)
        self.params["nb_epoch"] = 10
        # TODO: confirm tensor ordering in mxnet
        # self.params["channels_first"] = True

    def run(self):
        x_train, y_train = self.load_data()

        mod = importlib.import_module("benchmarker.modules.problems." +
                                      self.params["problem"]["name"] + ".mxnet")
        get_model = getattr(mod, 'get_model')

        net = get_model(x_train[0].shape)
        if self.params["nb_gpus"] > 0:
            devices = [mx.gpu(i) for i in self.params["gpus"]]
        else:
            devices = mx.cpu()
        if self.params["nb_gpus"] == 1:
            devices = mx.gpu(self.params["gpus"][0])

        print(devices)
        nb_epoch = 3
        train_iter = mx.io.NDArrayIter(x_train, y_train, batch_size=self.params["batch_size"])
        mod = mx.mod.Module(symbol=net,
                            # context=mx.cpu(),
                            context=devices,
                            data_names=['data'],
                            label_names=['softmax_label'])
        print("preheat")
        mod.fit(train_iter,
                # eval_data=data.get_iter(batch_size),
                optimizer='sgd',
                optimizer_params={'learning_rate': 0.1},
                eval_metric='acc',
                num_epoch=1)
        print("train")
        start = timer()
        mod.fit(train_iter,
                # eval_data=data.get_iter(batch_size),
                optimizer='sgd',
                optimizer_params={'learning_rate': 0.1},
                eval_metric='acc',
                num_epoch=nb_epoch)
        end = timer()
        self.params["time"] = (end-start) / nb_epoch
        self.params["framework_full"] = "MXNet-" + mx.__version__
        return self.params


def run(params):
    mxnet_backend = DoMxnet(params)
    return mxnet_backend.run()
