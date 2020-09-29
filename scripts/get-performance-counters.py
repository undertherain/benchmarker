#!/usr/bin/env python

import os, re
import sys
import glob

basedir = '../flops'

models = ['conv2d']
devices = ['perf']

#FP_ARITH:SCALAR_DOUBLE: Number of scalar double precision floating-point arithmetic instructions (multiply by 1 to get flops)
r5301c7_re  = re.compile('([\d|\,]+)\s+r5301c7([\(|\+|\-|\s]+)[\d|\.]+')

#SCALAR_SINGLE: Number of scalar single precision floating-point arithmetic instructions (multiply by 1 to get flops)
r5302c7_re = re.compile('([\d|\,]+)\s+r5302c7([\(|\+|\-|\s]+)[\d|\.]+')

#FP_ARITH:128B_PACKED_DOUBLE: Number of scalar 128-bit packed double precision floating-point arithmetic instructions (multiply by 2 to get flops)
r5304c7_re = re.compile('([\d|\,]+)\s+r5304c7([\(|\+|\-|\s]+)[\d|\.]+')

#FP_ARITH:128B_PACKED_SINGLE:Number of scalar 128-bit packed single precision floating-point arithmetic instructions (multiply by 4 to get flops)
r5308c7_re = re.compile('([\d|\,]+)\s+r5308c7([\(|\+|\-|\s]+)[\d|\.]+')

#FP_ARITH:256B_PACKED_DOUBLE:Number of scalar 256-bit packed double precision floating-point arithmetic instructions (multiply by 4 to get flops)
r5310c7_re = re.compile('([\d|\,]+)\s+r5310c7([\(|\+|\-|\s]+)[\d|\.]+')

#FP_ARITH:256B_PACKED_SINGLE:Number of scalar 256-bit packed single precision floating-point arithmetic instructions (multiply by 8 to get flops)
r5320c7_re = re.compile('([\d|\,]+)\s+r5320c7([\(|\+|\-|\s]+)[\d|\.]+')

time_re = re.compile('\"time_total"\:\s+[\d|\.]+')

#nvprof output
fp1_re = re.compile('\d+\s+flop\_count\_sp\s+Floating\s+Point\s+Operations\(Single Precision\)\s+\d+\.\d+e[\+|\-]\d+\s+\d+\.\d+e[\+|\-]\d+\s+\d+\.\d+e[\+|\-]\d+')

#nvprof output
fp2_re = re.compile('\\d+\s+flop\_count\_sp\s+Floating\s+Point\s+Operations\(Single Precision\)\s+\d+\s+\d+\s+\d+')

#estimated operations from benchmarker
flop_est_re = re.compile('\"flop_estimated"\:\s+[\d]+')

print('Model Device File Total_Time Estimated_Operations r5302c7 r5308c7 r5320c7 /nvprof_fp')
print("------------------------------------------------------------------------------------")

for model in models:
                
	currdir = os.path.join(basedir, model)

	for device in devices:
		dname = os.path.join(currdir, device)
		if not os.path.exists(dname):
			print('ERROR: Cannot find directory: ', dname)
			sys.exit(1)
		
		files = glob.glob(dname + "/*") 
		for file in files:
			r5301c7 = r5302c7 = r5304c7 = r5308c7 = r5310c7 = r5320c7 = nvprof_fp = 0
			head, tail = os.path.split(file) 
			runfile = open(file, 'r')
			time_match = 0
			nvidia_fp = False
			for line in runfile:
				if r5301c7_re.search(line):
					match_list = []
					match_list = r5301c7_re.search(line).group().split()
					#print(match_list)
					r5301c7 = match_list[0] 
			
				if r5302c7_re.search(line):
					match_list = []
					match_list = r5302c7_re.search(line).group().split()
					#print(match_list)
					r5302c7 = match_list[0] 

				if r5304c7_re.search(line):
					match_list = []
					match_list = r5304c7_re.search(line).group().split()
					#print(match_list)
					r5304c7 = match_list[0]

				if r5308c7_re.search(line):
					match_list = []
					match_list = r5308c7_re.search(line).group().split()
					#print(match_list)
					r5308c7 = match_list[0]

				if r5310c7_re.search(line):
					match_list = []
					match_list = r5310c7_re.search(line).group().split()
					#print(match_list)
					r5310c7 = match_list[0]

				if r5320c7_re.search(line):
					match_list = []
					match_list = r5320c7_re.search(line).group().split()
					#print(match_list)
					r5320c7 = match_list[0]
 
				if time_re.search(line) and time_match == 0:
					match_list = []
					match_list = time_re.search(line).group().split()
					#print(match_list)
					run_time = match_list[1]
					time_match = 1

				if flop_est_re.search(line):
                                        match_list = []
                                        match_list = flop_est_re.search(line).group().split()
                                        #print(match_list)
                                        flop_estimated = match_list[1]


				if fp2_re.search(line) or fp1_re.search(line):
					match_list = []
					if fp2_re.search(line):
						match_list = fp2_re.search(line).group().split()
					else:
						match_list = fp1_re.search(line).group().split()
						match_list[8] = int(float(match_list[8]))
					nvprof_fp += int(match_list[0]) * int(match_list[8])
					nvidia_fp = True

			runfile.close()
			if nvidia_fp:
				print(model, device, tail, run_time, flop_estimated, nvprof_fp)
			else:
				print(model, device, tail, run_time, flop_estimated, r5302c7, r5308c7, r5320c7)
                            
print("done")               
