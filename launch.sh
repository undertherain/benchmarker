#!/bin/sh

problem=--problem=conv2d_2

python3 benchmarker.py --framework=theano ${problem}
python3 benchmarker.py --framework=tensorflow ${problem}
python3 benchmarker.py --framework=chainer ${problem}
python3 benchmarker.py --framework=mxnet ${problem}
