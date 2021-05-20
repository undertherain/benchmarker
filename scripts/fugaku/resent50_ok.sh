#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=eap-small"
#PJM -L elapse=00:30:00
#PJM -L "node=1"
#PJM -j
#PJM -S

export PATH=/home/apps/oss/PyTorch-1.7.0/bin:$PATH
export LD_LIBRARY_PATH=/home/apps/oss/PyTorch-1.7.0/lib:$LD_LIBRARY_PATH

# sps: 10
LD_PRELOAD=libtcmalloc.so OMP_NUM_THREADS=12 numactl -N 4 -m 4 python3 -m benchmarker --framework=pytorch --problem=bert --problem_size=120,128 --batch_size=24 --nb_epoch=5 --mode=training --backend=DNNL --tensor_layout=DNNL
