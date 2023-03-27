# CPU:

python3 -m benchmarker.benchmarker \
    --framework=pytorch \
    --problem=cosmoflow \
    --problem_size=128,4,256,256,256 \
    --batch_size=16 \
    --nb_epoch=2 \
    --gpus=0 \
    --preheat \
    --mode=inference \
    --precision=TF32

#    --flops \
