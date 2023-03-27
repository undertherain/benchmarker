python3 -m benchmarker \
    --framework=pytorch \
    --problem=deepcam \
    --problem_size=60 \
    --batch_size=6 \
    --mode=inference \
    --gpus=0 \
    --nb_epoch=1
