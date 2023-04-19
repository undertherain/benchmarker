#!/bin/bash
#SBATCH --account=project_462000123
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=2:00:00

# --cpus-per-task=32
module load cray-python

BASE_ARGS=(-m benchmarker.benchmarker --framework=pytorch)
BERT_ARGS=(
    --problem=roberta_large_mlm
    --problem_size=512,256
    --batch_size=32
    --nb_epoch=2
    --gpus=0
    --preheat
    --mode=inference
)

COSMOFLOW_ARGS=(
    --problem=cosmoflow
    --problem_size=128
    --input_shape=4,256,256,256
    --batch_size=32
    --nb_epoch=2
    --gpus=0
    --preheat
    --mode=inference
)

DEEPCAM_ARGS=(
    --problem=deepcam
    --problem_size=320,3,512,512
    --batch_size=64
    --preheat
    --mode=inference
    --gpus=0
    --nb_epoch=5
)

LSTM_ARGS=(
    --problem=lstm
    --problem_size=256,1024,768
    --bidirectional=True
    --batch_size=64
    --mode=inference
    --cnt_layers=8
    --gpus=0
    --preheat
    --nb_epoch=4
    --flops
)

for p in FP16 FP32 TF32; do
    echo ">>>> 1/4 BERT ($p) <<<<"
    python3 ${BASE_ARGS[@]} ${BERT_ARGS[@]} --precision=$p
    # echo ">>>> 2/4 COSMOFLOW ($p) <<<<"
    # python3 ${BASE_ARGS[@]} ${COSMOFLOW_ARGS[@]} --precision=$p
    echo ">>>> 3/4 DEEPCAM ($p) <<<<"
    python3 ${BASE_ARGS[@]} ${DEEPCAM_ARGS[@]} --precision=$p
    echo ">>>> 4/4 LSTM ($p) <<<<"
    python3 ${BASE_ARGS[@]} ${LSTM_ARGS[@]} --precision=$p
done
