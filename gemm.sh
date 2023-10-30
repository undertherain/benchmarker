LD_PRELOAD=libtcmalloc.so \
python3 -m benchmarker.benchmarker \
    --framework=torch \
    --problem=gemm \
    --problem_size=16000,16000,16000 \
    --nb_epoch=10 \
    --precision=FP32 \
    --preheat
#    --batch_size=1536 \
#    --gpus=0 \
#    --enable_TF32    


