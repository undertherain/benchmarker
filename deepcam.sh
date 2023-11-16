python3 -m benchmarker \
    --framework=pytorch \
    --backend=native \
    --problem=deepcam \
    --problem_size=1,3,512,512 \
    --batch_size=1 \
    --mode=training \
    --nb_epoch=1 \
    --precision=FP32
