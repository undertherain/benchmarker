import importlib
import json
import os
import datetime
import begin
import sys
sys.path.append("kernels")
sys.path.append("data_helpers")
from sysinfo.cute_device import get_cute_device_str
from sysinfo import sysinfo 

def get_time_str():
    d = datetime.datetime.now()
    s = d.strftime("%y.%m.%d_%H.%M.%S")
    return s


def save_params(params):
	str_result=json.dumps(params, sort_keys=True,  indent=4, separators=(',', ': '))
	print (str_result)	
	path_out = "/work/alex/data/DL_perf/json"
#	path_out = "/tmp/dl"
	name_file = params["problem"]+ "_" +params["framework"]+"_"+params["device"] + "_" + get_time_str()+".json"
	with open(os.path.join(path_out,name_file),"w") as f:
		f.write(str_result)

@begin.start
def main(framework: "Framework to test" = "theano", problem: "problem to solve" = "conv2d_1"):
	print ("benchmarker")
	params=sysinfo.get_sys_info()
	params["framework"]=framework
	params["nb_gpus"]=1
	params["problem"]=problem
	params["batch_size"]=8

	mod = importlib.import_module("kernels.problems."+params["problem"]+".data")
	get_data = getattr(mod, 'get_data')
	data = get_data()

	try:
		params["bytes_x_train"] = data[0].nbytes    #todo move this to DL frameworks module
		params["shape_x_train"] = data[0].shape    
	except Exception as e:
		pass
	
	if params["nb_gpus"]>0:
		params["device"]=get_cute_device_str(params["gpu"])
	else:
		params["device"]=get_cute_device_str(params["cpu_brand"])


	#if params["framework"]=="tensorflow":
		#os.environ["KERAS_BACKEND"]="tensorflow"
		#if params["nb_gpus"]<1:
			#os.environ['CUDA_VISIBLE_DEVICES'] = ""   
		#if params["nb_gpus"]>1:
			#print ("multiple gpus with TF not supported yet")
			#return
		#from do_keras import run
#
	#if not params["framework"] in ["theano","tensorflow"]:
	mod = importlib.import_module("kernels.do_"+params["framework"])
	run = getattr(mod, 'run')

	params = run(params, data)
	save_params(params)
