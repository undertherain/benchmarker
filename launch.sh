#!/bin/sh
#/work/alex/data/DL_perf/json

problem=--problem=conv2d_1 
gpus=--gpus=1

for framework in "theano" "tensorflow" "chainer" "mxnet"
do
	for problem in "conv2d_1" "conv2d_2" "conv3d_1"
	do
        	python3 benchmarker.py --framework=$framework --problem=$problem $gpus --path-out=/work/alex/DL_perf/json
	done
done
