# CPU:
args=(
    -m benchmarker.benchmarker
    --framework=pytorch
    --problem=cosmoflow
    --problem_size=256
    --input_shape=4,256,256,256
    --batch_size=8
    --nb_epoch=10
    --gpus=0
    --preheat
    --mode=inference
)

echo $args \
    --numerics=FP32 \
    --preciosn=highest
