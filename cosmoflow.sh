# CPU:

python3 -m benchmarker.benchmarker \
    --framework=pytorch \
    --problem=cosmoflow \
    --problem_size=128 \
    --input_shape=4,256,256,256 \
    --batch_size=32 \
    --nb_epoch=2 \
    --gpus=0 \
    --preheat \
    --mode=inference \
    --precision=FP16

#    --flops \
