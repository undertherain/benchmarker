#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=01:00:00
#$ -N benchmarker-deepcam
#$ -j y
#$ -o $JOB_NAME.o$JOB_ID

# run with: qsub -g gcb50300 <this file>
source /etc/profile.d/modules.sh
module load gcc/9.3.0
module load python/3.8/3.8.7

python3 -m benchmarker \
        --gpus=0 \
        --framework=pytorch \
        --problem=deepcam \
        --problem_size=160 \
        --batch_size=32
