LD_PRELOAD=libtcmalloc.so \
python3 -m benchmarker.benchmarker \
    --framework=pytorch \
    --problem=adam \
    --model_size=100000000    \
    --nb_epoch=20 \
    --numerics=FP32 \
    --precision=highest \
    --cnt_samples=1 \
    --batch_size=1 \
    --preheat \
    --mode=inference \
    --gpus=0 \
#    --batch_size=1536 \
#    --enable_TF32


