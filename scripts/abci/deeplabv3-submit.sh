#!/bin/bash
#$ -cwd
#$ -l rt_F=8
#$ -l h_rt=07:55:00
#$ -N benchmarker-deeplabv3
#$ -j y
#$ -o $JOB_NAME.o$JOB_ID

# run with: qsub -g gcb50300 <this file>
source /etc/profile.d/modules.sh
module load gcc/9.3.0
module load python/3.8/3.8.7
python3 -m benchmarker.benchmarker --framework=pytorch --problem=deeplabv3_resnet50 --problem_size=32
