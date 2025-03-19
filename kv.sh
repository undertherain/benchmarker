LD_PRELOAD=libtcmalloc.so \
python3 -m benchmarker.benchmarker \
    --framework=pytorch \
    --problem=kv \
    --embedding_size=128    \
    --sequence_length=1000    \
    --nb_epoch=10 \
    --numerics=FP16 \
    --precision=highest \
    --cnt_samples=32 \
    --batch_size=32 \
    --preheat \
    --mode=inference \
    --gpus=0 \
#    --batch_size=1536 \
#    --enable_TF32


