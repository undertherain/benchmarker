#!/usr/bin/env python

import os, re
import sys
import glob

basedir = '../flops'

#Include kernels (conv2d, lstm) and models (resnet50, vgg16, bert etc.)
models = ['conv2d']

#Include devices (perf for cpu, nvidia for gpu)
devices = ['perf']

#Counters recorded by perf utility:
##SCALAR_SINGLE: Number of scalar single precision floating-point arithmetic instructions (multiply by 1 to get flops)
#FP_ARITH:128B_PACKED_SINGLE:Number of scalar 128-bit packed single precision floating-point arithmetic instructions (multiply by 4 to get flops)
#FP_ARITH:256B_PACKED_SINGLE:Number of scalar 256-bit packed single precision floating-point arithmetic instructions (multiply by 8 to get flops)

hw_event = ["r5302c7", "r5308c7", "r5320c7"]

time_re = re.compile('\"time_total"\:\s+[\d|\.]+')

#nvprof output type 1
nvprof_exp1_re = re.compile('\d+\s+flop\_count\_sp\s+Floating\s+Point\s+Operations\(Single Precision\)\s+\d+\.\d+e[\+|\-]\d+\s+\d+\.\d+e[\+|\-]\d+\s+\d+\.\d+e[\+|\-]\d+')

#nvprof output type 2
nvprof_exp2_re = re.compile('\\d+\s+flop\_count\_sp\s+Floating\s+Point\s+Operations\(Single Precision\)\s+\d+\s+\d+\s+\d+')

#estimated operations from benchmarker
flop_est_re = re.compile('\"flop_estimated"\:\s+[\d]+')

print('Device File Estimated Measured')
print("------------------------------")

for model in models:
                
	currdir = os.path.join(basedir, model)

	for device in devices:
		dname = os.path.join(currdir, device)
		if not os.path.exists(dname):
			print('ERROR: Cannot find directory: ', dname)
			sys.exit(1)
		
		files = glob.glob(dname + "/*") 
		for file in files:
			perf_events = {}
			nvprof_fp = time_match = 0
			head, tail = os.path.split(file) 
			runfile = open(file, 'r')
			nvidia_fp = False
			for line in runfile:
				for event in hw_event:
					match_exp = re.compile('[\d|\,]+\s+' + event).search(line)
					if match_exp:
						match_list = match_exp.group().split()
						perf_events[event] = int(match_list[0].replace(',',''))				

				if time_re.search(line) and time_match == 0:
					match_list = time_re.search(line).group().split()
					run_time = match_list[1]
					time_match = 1

				if flop_est_re.search(line):
                                        match_list = flop_est_re.search(line).group().split()
                                        gflop_estimated = int(match_list[1])/10 ** 9


				if nvprof_exp2_re.search(line) or nvprof_exp1_re.search(line):
					if nvprof_exp2_re.search(line):
						match_list = nvprof_exp2_re.search(line).group().split()
					else:
						match_list = nvprof_exp1_re.search(line).group().split()
						match_list[8] = int(float(match_list[8]))
					nvprof_fp += int(match_list[0]) * int(match_list[8])
					nvidia_fp = True

			runfile.close()
			gflop_measured = (perf_events['r5302c7'] + 4*perf_events['r5308c7'] + 8*perf_events['r5320c7'])/10**9
			if nvidia_fp:
				print(model, device, tail, run_time, flop_estimated, nvprof_fp)
			else:
				#print(model, device, tail, run_time, gflop_estimated, gflop_measured)
				print(device, tail, gflop_estimated, gflop_measured)
                            
