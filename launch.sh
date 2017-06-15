#!/bin/sh
set -o xtrace
#/work/alex/data/DL_perf/json

#problem=--problem=conv2d_1 
#gpus=--gpus=1

#for framework in "theano" "tensorflow" "chainer" "mxnet"
#for framework in  "mxnet" "chainer"
for framework in  "chainer" "mxnet"
do
	for problem in "conv2d_1" "conv2d_2" "conv3d_1" "vgg16"
	do
        for gpus in "0" "0,1" "0,1,2" "0,1,2,3" 
        do
        	python3 benchmarker.py --framework=$framework --problem=$problem --gpus=$gpus --path-out=/work/alex/data/DL_perf/json
        done
	done
done
