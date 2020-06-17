# -*- coding: utf-8 -*-
"""Chainer support.
"""

from timeit import default_timer as timer
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from .i_neural_net import INeuralNet
# import chainerx as chx


class Benchmark(INeuralNet):
    def __init__(self, params, remaining_args=None):
        super().__init__(params, remaining_args)
        self.params["channels_first"] = True

    def do_inference(self, model, x_train, y_train):
        chainer.enable_backprop = False
        print("doing inference")
        print("x_train:", x_train.shape)
        # TODO: move to GPU id needed
        if self.params["nb_gpus"] > 1:
            print("multi-gpu inference is not support")
            exit(-1)
        if self.params["nb_gpus"] == 1:
            print("movin data to gpu")
            import cupy
            cupy.cuda.Device(self.params["gpus"][0]).use()
            # TODO if in core
            # x_train = cupy.array(x_train)

        for id_epoch in range(self.params["nb_epoch"]):
            print("epoch ", id_epoch)
            for i in range(x_train.shape[0]):
                minibatch = x_train[i]
                _ = model.predictor(minibatch)

        # TODO: add iterator
        # iterate over all mini-batches

    def do_training(self, model, x_train, y_train):
        params = self.params
        # optimizer = chainer.optimizers.SGD()
        if params["nb_gpus"] == 1:
            import cupy
            id_device = params["gpus"][0]
            cupy.cuda.Device(id_device).use()
        optimizer = chainer.optimizers.MomentumSGD(lr=0.001, momentum=0.95)
        optimizer.setup(model)
        for id_epoch in range(self.params["nb_epoch"]):
            print("epoch ", id_epoch)
            for data, target in zip(x_train, y_train):
                if self.params["nb_gpus"] == 1:
                    # TODO: option for on-core training
                    data = cupy.array(data)
                    target = cupy.array(target)
                pred = model.predictor(data)
                loss = F.softmax_cross_entropy(pred, target)
                loss.backward()
        return

        # using Chainer's native iterators
        x_train = x_train.reshape((x_train.shape[0] * x_train.shape[1],) + x_train.shape[2:])
        y_train = y_train.reshape((y_train.shape[0] * y_train.shape[1],))
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

    def run_internal(self):
        # TODO set a config option to use ChainerX or other backend
        use_chainer_x = False
        params = self.params
        x_train, y_train = self.load_data()

        # if len(Y_train.shape) == 1:
        #     Y_train = Y_train[:, np.newaxis]
        #     model = Classifier(Net())
        # else:
        model = L.Classifier(self.net)
        # r = self.net(x_train[:1])
        # print(r.shape, r[0][:10])
        # exit(-1)
        if use_chainer_x:
            x_train = chx.array(x_train)
            y_train = chx.array(y_train)
            model.to_device('native:0')
        if params["nb_gpus"] == 1:
            id_device = params["gpus"][0]
            chainer.cuda.get_device(id_device).use()
            if use_chainer_x:
                model.to_device(f'cuda:{id_device}')
            else:
                model.to_gpu(self.params["gpus"][id_device])

        # print("X_train:", type(X_train), X_train.shape)
        # print("Y_train:", type(Y_train), Y_train.shape, Y_train[:10])
        # result = model.predictor(X_train)
        # print (result.shape)
        # TODO: pre-heat
        start = timer()
        if params["mode"] == "training":
            self.do_training(model, x_train, y_train)
        else:
            self.do_inference(model, x_train, y_train)
        end = timer()

        params["time"] = (end - start) / self.params["nb_epoch"]
        params["framework_full"] = "Chainer-" + chainer.__version__
        return params
