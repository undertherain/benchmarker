#!/bin/sh

problem=--problem=conv2d_1 
gpus=--gpus=1

python3 benchmarker.py --framework=theano ${problem} ${gpus}
python3 benchmarker.py --framework=tensorflow ${problem}  ${gpus}
python3 benchmarker.py --framework=chainer ${problem}  ${gpus}
python3 benchmarker.py --framework=mxnet ${problem} ${gpus}

problem=--problem=conv2d_2 

python3 benchmarker.py --framework=theano ${problem} ${gpus}
python3 benchmarker.py --framework=tensorflow ${problem} ${gpus}
python3 benchmarker.py --framework=chainer ${problem} ${gpus}
python3 benchmarker.py --framework=mxnet ${problem} ${gpus}

problem=--problem=conv3d_1 

python3 benchmarker.py --framework=theano ${problem} ${gpus}
python3 benchmarker.py --framework=tensorflow ${problem} ${gpus}
python3 benchmarker.py --framework=chainer ${problem} ${gpus}
python3 benchmarker.py --framework=mxnet ${problem} ${gpus}

#python3 benchmarker.py --framework=theano --problem=conv2d_1
#python3 benchmarker.py --framework=theano --problem=conv2d_2
#python3 benchmarker.py --framework=theano --problem=conv3d_1
