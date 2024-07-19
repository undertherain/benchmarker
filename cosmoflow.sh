# CPU:
args="
    -m benchmarker.benchmarker\
    --framework=pytorch\
    --problem=cosmoflow\
    --sample_shape=4,256,256,256\
    --cnt_samples=8\
    --batch_size=1\
    --nb_epoch=10\
    --gpus=0\
    --preheat\
    --mode=training"

python3 $args \
    --numerics=FP32 \
    --precision=highest
