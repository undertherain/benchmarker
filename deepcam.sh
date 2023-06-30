python3 -m benchmarker \
    --framework=pytorch \
    --backend=native \
    --problem=deepcam \
    --problem_size=48,3,512,512 \
    --batch_size=48 \
    --mode=inference \
    --nb_epoch=1 \
    --precision=FP32
