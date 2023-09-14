# CPU:
LD_PRELOAD=libtcmalloc.so \
python3 -m benchmarker.benchmarker \
    --framework=pytorch \
    --backend=native \
    --tensor_layout=native \
    --problem=roberta_large_mlm \
    --problem_size=32,256 \
    --batch_size=32 \
    --nb_epoch=1 \
    --preheat \
    --mode=inference \
    --precision=FP32 \
    --gpus=0


