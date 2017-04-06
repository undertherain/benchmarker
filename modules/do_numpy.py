import numpy as np
from timeit import default_timer as timer


def run(params,data):
	a=np.random.random((2048,2048))
	b=np.random.random((2048,2048))
	nb_epoch = 1

	start=timer()
	for i in range(nb_epoch):
		c = a @ b
	end=timer()

	#params["framework"]=="theano":
	params["time"]=(end-start)/nb_epoch
	return params
