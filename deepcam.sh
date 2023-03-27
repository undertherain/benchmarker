python3 -m benchmarker \
    --framework=pytorch \
    --problem=deepcam \
    --problem_size=60,3,512,512 \
    --batch_size=6 \
    --preheat \
    --mode=inference \
    --gpus=0 \
    --nb_epoch=2
