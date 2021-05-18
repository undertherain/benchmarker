#!/bin/sh

LD_PRELOAD=libtcmalloc.so OMP_NUM_THREADS=12 run_on_cmg python3 -m benchmarker --problem=resnet50 --framework=pytorch --problem_size=120 --batch_size=24 --nb_epoch=3 --mode=training --backend=DNNL --tensor_layout=DNNL