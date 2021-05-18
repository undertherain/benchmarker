#!/bin/bash

# sps: 10
LD_PRELOAD=libtcmalloc.so OMP_NUM_THREADS=12 numactl -N 4 -m 4 python3 -m benchmarker --framework=pytorch --problem=bert --problem_size=120,128 --batch_size=24 --nb_epoch=5 --mode=training --backend=DNNL --tensor_layout=DNNL
