#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=eap-small"
#PJM -L elapse=00:30:00
#PJM -L "node=1"
#PJM -j
#PJM -S

export PATH=/home/apps/oss/PyTorch-1.7.0/bin:$PATH
export LD_LIBRARY_PATH=/home/apps/oss/PyTorch-1.7.0/lib:$LD_LIBRARY_PATH

python3 -m benchmarker --framework=pytorch --problem=deeplabv3_resnet50 --problem_size=32 --nb_epoch=3 --power_rapl
