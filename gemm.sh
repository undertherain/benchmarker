LD_PRELOAD=libtcmalloc.so \
python3 -m benchmarker.benchmarker \
    --framework=torch \
    --problem=gemm \
    --sample_shape=16000,16000,16000 \
    --nb_epoch=10 \
    --precision=FP16 \
    --preheat \
    --gpus=0 \
#    --batch_size=1536 \
#    --enable_TF32    


