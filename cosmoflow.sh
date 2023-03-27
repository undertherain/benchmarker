# CPU:

python3 -m benchmarker.benchmarker \
    --framework=pytorch \
    --problem=cosmoflow \
    --problem_size=60 \
    --input_shape=4,128,128,128 \
    --batch_size=6 \
    --nb_epoch=2 \
    --gpus=0 \
    --preheat \
    --mode=inference \
    --precision=TF32

#    --flops \
