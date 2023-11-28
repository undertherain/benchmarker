# CPU:
#LD_PRELOAD=libtcmalloc.so \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
python3 -m benchmarker \
    --framework=pytorch \
    --backend=native \
    --tensor_layout=native \
    --problem=roberta_large_mlm \
    --problem_size=512,256 \
    --batch_size=64 \
    --nb_epoch=10 \
    --preheat \
    --mode=training \
    --precision=TF32 \
    --gpus=0 \
    --power_nvml
