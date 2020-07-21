#!/bin/sh
# HACK this does not work with pip-installed benchmarker
# because binaries are not in site_packages, alse can't use run script
# as temp solution - this sh script to be copied to root folder manually when needed
for framework in cblas torch numpy
do
    python3 -m benchmarker --problem=gemm --framework=$framework --problem_size=1024,1024,1024
    python3 -m benchmarker --problem=gemm --framework=$framework --problem_size=16000,16000,1600
    python3 -m benchmarker --problem=gemm --framework=$framework --problem_size=65000,27,32
done