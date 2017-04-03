#!/usr/bin/env python3

import platform
import cpuinfo
import json


def get_sys_info():
	#result ="getting system info"
	result = {}
	result["linux"] = platform.platform()
	result["linux"] = platform.platform()
	#get GPU, 

		#GPU mem, 
	#get CPU
	cpu = cpuinfo.get_cpu_info()
	result["cpu_brand"] = cpu["brand"]
	result["cpu_count"] = cpu["count"]
	result["hostname"] = platform.node()
	result["hostname_short"] = platform.node().split(".")[0]

	try:
		import pycuda.driver as cuda
		import pycuda.autoinit
		result["gpu"] = cuda.Device(0).name()
	except:
		result["gpu"] = "not detected"

	#get linux version
	#get kernel version

	return result

if 	__name__=="__main__":
	info = get_sys_info()
	print (json.dumps(info, sort_keys=True,  indent=4, separators=(',', ': ')))
	#print (get_sys_info())