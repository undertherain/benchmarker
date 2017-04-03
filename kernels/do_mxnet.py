import mxnet as mx

from timeit import default_timer as timer


def get_model(shape):
	net = mx.symbol.Variable('data')
	#c1 = mx.symbol.Convolution(data = data, name='conv1', kernel=(2, 2, 2), num_filter=16)
	net = mx.symbol.Convolution(data = net, name='conv1', kernel=(2, 2, 2), num_filter=16)
	net = mx.symbol.Activation(data = net, name='relu1', act_type="relu")
	#net = mx.symbol.Pooling(data = net,name ="mp1",kernel=(2,2,2),pool_type="max")
	#net = mx.symbol.Convolution(data = net, name='conv2', kernel=(2, 2, 2), num_filter=16)
	#net = mx.symbol.Activation(data = net, name='relu2', act_type="relu")
	#fc = mx.symbol.FullyConnected(data = mp2, num_hidden=1, name="fc")
	#net  = mx.symbol.LogisticRegressionOutput(data = fc, name = 'softmax')
	#net = mx.symbol.Flatten(data=net,name="flat")
	net  = mx.symbol.FullyConnected(data = net, name='fc3', num_hidden=1)
	net = mx.symbol.LogisticRegressionOutput(data = net, name = 'softmax')
	return net



def run(params,X_train,Y_train):
	net = get_model(X_train[0].shape)
	if params["nb_gpus"]>0:
		devices=[mx.gpu(i) for i in range(params["nb_gpus"])]
	else:
		devices=mx.cpu()
	if params["nb_gpus"]==1:
		devices=mx.gpu(0)

	print(devices)
	nb_epoch=2
	train_iter = mx.io.NDArrayIter(X_train, Y_train, batch_size=params["batch_size"])
	mod = mx.mod.Module(symbol=net,
		#context=mx.cpu(),
		context=devices,
		data_names=['data'],
		label_names=['softmax_label'])
	print("preheat")
	mod.fit(train_iter,
		#eval_data=data.get_iter(batch_size),
		optimizer='sgd',
		optimizer_params={'learning_rate':0.1},
		eval_metric='acc',
		num_epoch=1)
	print("train")
	start=timer()
	mod.fit(train_iter,
		#eval_data=data.get_iter(batch_size),
		optimizer='sgd',
		optimizer_params={'learning_rate':0.1},
		eval_metric='acc',
		num_epoch=nb_epoch)
	end=timer()
	params["time"]=(end-start)/nb_epoch
	params["framework_full"] = "MXNet-" + mx.__version__ 
	return params

