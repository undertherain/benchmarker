# CPU:

python3 -m benchmarker.benchmarker \
    --framework=pytorch \
    --problem=cosmoflow \
    --problem_size=2 \
    --batch_size=1 \
    --nb_epoch=1 \
    --gpus=0 \
    --mode=inference

#    --flops \
