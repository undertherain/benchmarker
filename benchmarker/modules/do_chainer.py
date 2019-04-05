# -*- coding: utf-8 -*-
"""Chainer support.
"""

import importlib
from timeit import default_timer as timer
import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions
from .i_neural_net import INeuralNet
# import chainerx as chx


class DoChainer(INeuralNet):
    def __init__(self, params):
        super().__init__(params)
        self.params["channels_first"] = True
        self.params["nb_epoch"] = 10

    def do_inference(self, model, x_train, y_train):
        chainer.enable_backprop = False
        print("doing inference")

    def do_training(self, model, x_train, y_train):
        params = self.params
        optimizer = chainer.optimizers.SGD()
        optimizer.setup(model)
        train = chainer.datasets.tuple_dataset.TupleDataset(x_train, y_train)
        # test  = chainer.datasets.tuple_dataset.TupleDataset(X_test,Y_test)
        if params["nb_gpus"] == 0:
            train_iter = chainer.iterators.SerialIterator(train, batch_size=params["batch_size"], repeat=True, shuffle=False)
        else:
            train_iter = chainer.iterators.MultiprocessIterator(train, batch_size=params["batch_size"], repeat=True, shuffle=True, n_processes=4)
            # train_iter = chainer.iterators.SerialIterator(train, batch_size=params["batch_size"], repeat=True, shuffle=False)
        # test_iter = chainer.iterators.SerialIterator(test, batch_size=batch_size=params["batch_size"], repeat=False, shuffle=False)
        if params["nb_gpus"] == 0:
            updater = training.StandardUpdater(train_iter, optimizer)
        else:
            if params["nb_gpus"] == 1:
                updater = training.StandardUpdater(train_iter, optimizer, device=id_device)
            else:
                dic_devices = {str(i): i for i in params["gpus"][1:]}
                dic_devices["main"] = params["gpus"][0]
                updater = training.ParallelUpdater(train_iter, optimizer, devices=dic_devices)

        trainer = training.Trainer(updater, (self.params["nb_epoch"], 'epoch'), out='/tmp/result')
        # trainer.extend(extensions.Evaluator(test_iter, model, device=id_device))
        # trainer.extend(extensions.Evaluator(test_iter, model))
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', "elapsed_time"]))
        trainer.run()

    def run(self):
        # TODO set a config option to use ChainerX or other backend
        use_chainer_x = False
        params = self.params
        x_train, y_train = self.load_data()
        mod = importlib.import_module("benchmarker.modules.problems." +
                                      params["problem"]["name"] + ".chainer")
        Net = getattr(mod, 'Net')
        # if len(Y_train.shape) == 1:
        #     Y_train = Y_train[:, np.newaxis]
        #     model = Classifier(Net())
        # else:
        model = L.Classifier(Net())
        if use_chainer_x:
            x_train = chx.array(x_train)
            y_train = chx.array(y_train)
            model.to_device('native:0')
        if params["nb_gpus"] == 1:
            id_device = params["gpus"][0]
            chainer.cuda.get_device(id_device).use()
            if use_chainer_x:
                model.to_device('cuda:0')
            else:
                model.to_gpu()

        # print("X_train:", type(X_train), X_train.shape)
        # print("Y_train:", type(Y_train), Y_train.shape, Y_train[:10])
        # result = model.predictor(X_train)
        # print (result.shape)

        start = timer()
        if params["mode"] == "training":
            self.do_training(model, x_train, y_train)
        else:
            self.do_inference(model, x_train, y_train)
        end = timer()

        params["time"] = (end - start) / self.params["nb_epoch"]
        params["framework_full"] = "Chainer-" + chainer.__version__
        return params


def run(params):
    backend_chainer = DoChainer(params)
    return backend_chainer.run()
