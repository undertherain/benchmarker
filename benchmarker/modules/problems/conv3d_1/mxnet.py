import mxnet as mx


def get_model(shape):
    net = mx.symbol.Variable('data')
    net = mx.symbol.Convolution(data=net, name='conv1', kernel=(2, 2, 2), num_filter=32)
    net = mx.symbol.Activation(data=net, name='relu1', act_type="relu")
    net = mx.symbol.Convolution(data=net, name='conv2', kernel=(2, 2, 2), num_filter=32)
    net = mx.symbol.Activation(data=net, name='relu2', act_type="relu")
    net = mx.symbol.Flatten(data=net, name="flat")
    net  = mx.symbol.FullyConnected(data = net, name='fc', num_hidden=1)
    net = mx.symbol.LogisticRegressionOutput(data = net, name = 'softmax')
    return net
