#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=small"
#PJM -L elapse=01:00:00
#PJM -L "node=1"
#PJM -j
#PJM -S

export PATH=/home/apps/oss/PyTorch-1.7.0/bin:$PATH
export LD_LIBRARY_PATH=/home/apps/oss/PyTorch-1.7.0/lib:$LD_LIBRARY_PATH

python -m benchmarker \
       --framework=pytorch \
       --problem=deepcam \
       --problem_size=120 \
       --batch_size=12
