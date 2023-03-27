python3 -m benchmarker \
    --framework=pytorch \
    --problem=deepcam \
    --problem_size=320,3,512,512 \
    --batch_size=64 \
    --preheat \
    --mode=inference \
    --gpus=0 \
    --nb_epoch=5 \
    --precision=FP16
