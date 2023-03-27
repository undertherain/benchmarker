# CPU:

python3 -m benchmarker.benchmarker \
    --framework=pytorch \
    --problem=roberta_large_mlm \
    --problem_size=64,256 \
    --batch_size=32 \
    --nb_epoch=1 \
    --gpus=0 \
    --mode=inference

#    --flops \
