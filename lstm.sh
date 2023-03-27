python3 -m benchmarker \
    --framework=pytorch \
    --problem=lstm \
    --problem_size=256,1024,768 \
    --bidirectional=True \
    --batch_size=64 \
    --mode=inference \
    --cnt_layers=8 \
    --gpus=0 \
    --preheat \
    --nb_epoch=4 \
    --precision=FP32 \
    --flops

#    --profile
#    --profile_pytorch
