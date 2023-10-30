# CPU:
#LD_PRELOAD=libtcmalloc.so \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
python3 -m benchmarker.benchmarker \
    --framework=pytorch \
    --backend=native \
    --tensor_layout=native \
    --problem=roberta_large_mlm \
    --problem_size=6,256 \
    --batch_size=6 \
    --nb_epoch=3 \
    --preheat \
    --mode=training \
    --precision=FP32 \
    --gpus=0 \


