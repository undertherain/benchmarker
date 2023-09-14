# CPU:
LD_PRELOAD=libtcmalloc.so \
python3 -m benchmarker.benchmarker \
    --framework=pytorch \
    --backend=DNNL \
    --tensor_layout=DNNL \
    --problem=roberta_large_mlm \
    --problem_size=32,256 \
    --batch_size=32 \
    --nb_epoch=1 \
    --preheat \
    --mode=inference \
    --precision=FP32


