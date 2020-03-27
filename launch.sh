#!/bin/bash
set -o xtrace
#/work/alex/data/DL_perf/json

#problem=--problem=conv2d_1 
#gpus=--gpus=1

#for framework in "theano" "tensorflow" "chainer" "mxnet"
#for framework in  "mxnet" "chainer"
#for framework in  "pytorch" 
#do
#   for problem in "resnet50"
#   do
#        for gpus in "0" "0,1" "0,1,2" "0,1,2,3" 
#        do
#           python3 benchmarker.py --framework=$framework --problem=$problem --problem_size
#        done
#   done
#done
backend=native
for i in {16..16}
do
    size=$((i * 4))
    echo $i $size
    python3 -m benchmarker \
        --framework=pytorch \
        --mode=inference \
        --backend=$backend \
        --problem=conv2d \
        --batch_size=$i \
        --problem_size=$size,3,224,224 \
    	--size_kernel=3 \
    	--stride=1 \
	    --gpus=0
done
