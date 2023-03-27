python3 -m benchmarker.benchmarker \
    --framework=pytorch \
    --problem=lstm \
    --problem_size=256,512,768 \
    --bidirectional=True \
    --batch_size=32 \
    --mode=inference \
    --cnt_layers=6 \
    --gpus=0 \
#    --profile
#    --profile_pytorch
