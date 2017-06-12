import numpy as np
import chainer
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from timeit import default_timer as timer
import importlib
from i_neural_net import INeuralNet


class Classifier(chainer.Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.sigmoid_cross_entropy(y, t)
        accuracy = F.binary_accuracy(y, t)
        chainer.report({'loss': loss, 'accuracy': accuracy}, self)
        return loss


class DoChainer(INeuralNet):

    def run(self):
        params = self.params
#        if params["nb_gpus"] > 1:
#            raise Exception("Multiple GPUs with chainer not supported yet")
        X_train, Y_train = self.load_data()
        # print(Y_train.shape)
        nb_epoch = 10

        mod = importlib.import_module("problems."+params["problem"]+".chainer")
        Net = getattr(mod, 'Net')
        model = Classifier(Net())
        if params["nb_gpus"] == 1:
            id_device = params["gpus"][0]
            chainer.cuda.get_device(id_device).use()
            model.to_gpu()

        optimizer = chainer.optimizers.SGD()
        optimizer.setup(model)
        train = chainer.datasets.tuple_dataset.TupleDataset(X_train, Y_train[:, np.newaxis])
        # test  = chainer.datasets.tuple_dataset.TupleDataset(X_test,Y_test)
        train_iter = chainer.iterators.SerialIterator(train, batch_size=params["batch_size"], repeat=True, shuffle=False)
        # test_iter  = chainer.iterators.SerialIterator(test, batch_size=32, repeat=False, shuffle=False)

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
        trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy']))
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
