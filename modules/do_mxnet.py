import mxnet as mx
import importlib
from timeit import default_timer as timer
from i_neural_net import INeuralNet


class DoMxnet(INeuralNet):

    def run(self):
        X_train, Y_train = self.load_data()
    
        mod = importlib.import_module("problems." + self.params["problem"] + ".mxnet")
        get_model = getattr(mod, 'get_model')

        net = get_model(X_train[0].shape)
        if self.params["nb_gpus"] > 0:
            devices=[mx.gpu(i) for i in range(self.params["nb_gpus"])]
        else:
            devices = mx.cpu()
        if self.params["nb_gpus"] == 1:
            devices = mx.gpu(0)

        print(devices)
        nb_epoch = 3
        train_iter = mx.io.NDArrayIter(X_train, Y_train, batch_size=self.params["batch_size"])
        mod = mx.mod.Module(symbol=net,
            #context=mx.cpu(),
            context = devices,
            data_names = ['data'],
            label_names = ['softmax_label'])
        print("preheat")
        mod.fit(train_iter,
            #eval_data=data.get_iter(batch_size),
            optimizer = 'sgd',
            optimizer_params = {'learning_rate' : 0.1},
            eval_metric = 'acc',
            num_epoch = 1)
        print("train")
        start=timer()
        mod.fit(train_iter,
            #eval_data=data.get_iter(batch_size),
            optimizer = 'sgd',
            optimizer_params = {'learning_rate':0.1},
            eval_metric = 'acc',
            num_epoch = nb_epoch)
        end = timer()
        self.params["time"] = (end-start) / nb_epoch
        self.params["framework_full"] = "MXNet-" + mx.__version__ 
        return self.params


def run(params):
    m = DoMxnet(params)
    return m.run()
