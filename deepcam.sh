python3 -m benchmarker \
    --framework=pytorch \
    --backend=native \
    --problem=deepcam \
    --cnt_samples_per_epoch=64 \
    --sample_shape=3,512,512 \
    --batch_size=8 \
    --mode=training \
    --preheat \
    --nb_epoch=10 \
    --numerics=FP16 \
    --precision=medium \
    --gpus=0 \
