from keras.optimizers import SGD
from timeit import default_timer as timer
import keras 
import keras.models
from keras.models import Sequential, Model
import importlib

def run(params,data):
	X_train,Y_train = data
	#with open() as f:
		#str_model=f.read()
	#exec(str_model)
	#path_net="./nets/keras/"+params["net"]+".py"
	#from path_net import get_model
	mod = importlib.import_module("problems."+params["problem"]+".keras")
	get_model = getattr(mod, 'get_model')

	model = get_model(X_train[0].shape)
	optimizer=keras.optimizers.Adam()
	model.compile(loss = 'binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])
	print("preheat")
	model.fit(X_train, Y_train, batch_size=params["batch_size"], epochs=1)
	nb_epoch=3
	print("train")
	start=timer()
	model.fit(X_train, Y_train, batch_size=params["batch_size"], nb_epoch=nb_epoch, verbose=1)
	end=timer()
	params["time"]=(end-start)/nb_epoch
	if params["framework"]=="theano":
		import theano
		version_backend=theano.__version__
	else:
		import tensorflow as tf
		version_backend=tf.__version__
	params["framework_full"] = "Keras-" + keras.__version__ + "/" + keras.backend.backend() + "_"+version_backend
	return params

