import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from timeit import default_timer as timer
import importlib
from .i_neural_net import INeuralNet


#class Classifier(chainer.Chain):
#    def __init__(self, predictor):
#        super(Classifier, self).__init__(predictor=predictor)#
#
#    def __call__(self, x, t):
#        y = self.predictor(x)
#        loss = F.sigmoid_cross_entropy(y, t)
#        accuracy = F.binary_accuracy(y, t)
#        chainer.report({'loss': loss, 'accuracy': accuracy}, self)
#        return loss


class DoChainer(INeuralNet):
    def __init__(self, params):
        super().__init__(params)
        self.params["channels_first"] = True

    def run(self):
        params = self.params
        X_train, Y_train = self.load_data()
        nb_epoch = 10


        mod = importlib.import_module("benchmarker.modules.problems." + params["problem"]["name"] + ".chainer")
        Net = getattr(mod, 'Net')
        #if len(Y_train.shape) == 1:
        #    Y_train = Y_train[:, np.newaxis]
        #    model = Classifier(Net())
        #else:
        model = L.Classifier(Net())
        if params["nb_gpus"] == 1:
            id_device = params["gpus"][0]
            chainer.cuda.get_device(id_device).use()
            model.to_gpu()

        # print("X_train:", type(X_train), X_train.shape)
        # print("Y_train:", type(Y_train), Y_train.shape, Y_train[:10])
        # result = model.predictor(X_train)
        # print (result.shape)
        # return

        optimizer = chainer.optimizers.SGD()
        optimizer.setup(model)
        train = chainer.datasets.tuple_dataset.TupleDataset(X_train, Y_train)
        # test  = chainer.datasets.tuple_dataset.TupleDataset(X_test,Y_test)
        if params["nb_gpus"] == 0:
            train_iter = chainer.iterators.SerialIterator(train, batch_size=params["batch_size"], repeat=True, shuffle=False)
        else:
            train_iter = chainer.iterators.MultiprocessIterator(train, batch_size=params["batch_size"], repeat=True, shuffle=True, n_processes=4)
            #train_iter = chainer.iterators.SerialIterator(train, batch_size=params["batch_size"], repeat=True, shuffle=False)
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

        trainer = training.Trainer(updater, (nb_epoch, 'epoch'), out='/tmp/result')
        # trainer.extend(extensions.Evaluator(test_iter, model, device=id_device))
        # trainer.extend(extensions.Evaluator(test_iter, model))
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', "elapsed_time"]))
        # trainer.extend(extensions.ProgressBar())
        start = timer()
        trainer.run()
        end = timer()

        params["time"] = (end-start) / nb_epoch
        params["framework_full"] = "Chainer-" + chainer.__version__
        return params


def run(params):
    m = DoChainer(params)
    return m.run()
