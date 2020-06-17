# -*- coding: utf-8 -*-
"""MxNet support.

This module integrates MxNet framework into the benchmarker
"""
from timeit import default_timer as timer
import mxnet as mx
from .i_neural_net import INeuralNet


class Benchmark(INeuralNet):

    def __init__(self, params, unparsed_args):
        super().__init__(params, unparsed_args)
        # TODO: confirm tensor ordering in mxnet
        # self.params["channels_first"] = True

    def run_internal(self):
        params = self.params
        x_train, y_train = self.load_data()

        data = mx.sym.var('data')
        # if opt.dtype == 'float16':
        #     data = mx.sym.Cast(data=data, dtype=np.float16)
        out = self.net(data)
        # if opt.dtype == 'float16':
        #     out = mx.sym.Cast(data=out, dtype=np.float32)
        softmax = mx.sym.SoftmaxOutput(out, name='softmax')
        if self.params["nb_gpus"] > 0:
            devices = [mx.gpu(i) for i in self.params["gpus"]]
        else:
            devices = mx.cpu()
        if self.params["nb_gpus"] == 1:
            devices = mx.gpu(self.params["gpus"][0])

        mod = mx.mod.Module(softmax, context=devices)
#         train_data, val_data = get_data_iters(dataset, batch_size, opt)
        train_iter = mx.io.NDArrayIter(x_train, y_train, batch_size=self.params["batch_size"])

        print("preheat")
        mod.fit(train_iter,
                # eval_data=val_data,
                num_epoch=1,
                # kvstore=kv,
                # batch_end_callback=mx.callback.Speedometer(params[]batch_size, max(1, opt.log_interval)),
                # epoch_end_callback=mx.callback.do_checkpoint('image-classifier-%s'% opt.model),
                optimizer='sgd',
                optimizer_params={'learning_rate': 0.1},
                # optimizer_params={'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum, 'multi_precision': True},
                initializer=mx.init.Xavier(magnitude=2))

        # print(devices)
#        mod = mx.mod.Module(symbol=net,
#                            # context=mx.cpu(),
#                            context=devices,
#                            data_names=['data'],
#                            label_names=['softmax_label'])
        print("train")
        start = timer()
        mod.fit(train_iter,
                # eval_data=data.get_iter(batch_size),
                optimizer='sgd',
                optimizer_params={'learning_rate': 0.1},
                eval_metric='acc',
                num_epoch=params["nb_epoch"])
        end = timer()
        self.params["time"] = (end - start) / params["nb_epoch"]
        self.params["time_epoch"] = (end - start) / params["nb_epoch"]
        self.params["framework_full"] = "MXNet-" + mx.__version__
        return self.params
