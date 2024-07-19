python3 -m benchmarker \
    --framework=pytorch \
    --problem=lstm \
    --cnt_samples_per_epoch=256 \
    --sample_shape=1024,768 \
    --bidirectional=True \
    --batch_size=64 \
    --mode=inference \
    --cnt_layers=8 \
    --preheat \
    --nb_epoch=4 \
    --precision=FP32 \
    --gpus=0 \
    --power_nvml \
    

#    --profile
#    --profile_pytorch
