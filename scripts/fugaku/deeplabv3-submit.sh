#!/bin/bash

export PATH=/home/apps/oss/PyTorch-1.7.0/bin:$PATH
export LD_LIBRARY_PATH=/home/apps/oss/PyTorch-1.7.0/lib:$LD_LIBRARY_PATH

python3 -m benchmarker --framework=pytorch --problem=deeplabv3_resnet50 --problem_size=32 --nb_epoch=3 --power_rapl
