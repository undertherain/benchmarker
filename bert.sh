# CPU:

python3 -m benchmarker \
    --framework=pytorch \
    --problem=roberta_large_mlm \
    --problem_size=32,256 \
    --batch_size=32 \
    --nb_epoch=1 \
    --gpus=0 \
    --preheat \
    --mode=inference \
    --precision=FP16 \

#    --flops \
