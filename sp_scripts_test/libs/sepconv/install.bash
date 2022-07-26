#!/bin/bash

#
# Created by: Simon Niklaus, Long Mai, Feng Liu
# https://github.com/sniklaus/pytorch-sepconv
#

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"
echo "$parent_path"

TORCH=$(python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))")
echo ${TORCH}
GPU_ARCH="compute_37"

nvcc -ccbin gcc -c -o src/SeparableConvolution_kernel.o src/SeparableConvolution_kernel.cu --gpu-architecture=${GPU_ARCH} --gpu-code=${GPU_ARCH} --compiler-options -fPIC -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC

python3 install.py
