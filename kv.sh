LD_PRELOAD=libtcmalloc.so \
python3 -m benchmarker.benchmarker \
    --framework=pytorch \
    --problem=kv \
    --embedding_size=8000    \
    --nb_epoch=10 \
    --numerics=FP16 \
    --precision=highest \
    --cnt_samples=32 \
    --batch_size=1 \
    --preheat \
    --mode=inference
    # --gpus=0 \
#    --batch_size=1536 \
#    --enable_TF32    


