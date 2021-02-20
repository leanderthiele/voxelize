#!/bin/sh

export VOXELIZE=.

source ${VOXELIZE}/paths.sh

export NETWORK_PATH=${VOXELIZE}/data/network.pt
export VALIDATION_LOSS_PATH=${VOXELIZE}/data/validation_loss.bin
export INPUTS_PATH=${VOXELIZE}/data/inputs.bin
export OUTPUTS_PATH=${VOXELIZE}/data/outputs.bin

mkdir -p data

g++ -std=c++17 -O3 -g3 \
    -Wall -Wextra -Wunused-result -Wno-unused-parameter \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    -UCPU_ONLY \
    -DNETWORK_PATH=\"${NETWORK_PATH}\" \
    -DVALIDATION_LOSS_PATH=\"${VALIDATION_LOSS_PATH}\" \
    -DINPUTS_PATH=\"${INPUTS_PATH}\" \
    -DOUTPUTS_PATH=\"${OUTPUTS_PATH}\" \
    -I${TORCH}/include/torch/csrc/api/include/ \
    -I${TORCH}/include \
    -I${CUDA}/include \
    -I${CUDNN}/include \
    -I${VOXELIZE}/include \
    -o train_network_gpu train_network.cpp \
    -L${TORCH}/lib \
    -L${CUDA}/lib64 \
    -L${CUDNN}/lib64 \
    -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lcudart
