#!/bin/sh

export TORCH=${HOME}/pytorch-install
export CUDNN=/usr/local/cudnn/cuda-11.0/8.0.2
export CUDA=/usr/local/cuda-11.0

export VOXELIZE=.

mkdir -p ${VOXELIZE}/build

g++ -std=c++17 -O3 -g3 \
    -Wall -Wextra -Wunused-result -Wno-unused-parameter \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    -UCPU_ONLY \
    -I${TORCH}/include/torch/csrc/api/include/ \
    -I${TORCH}/include \
    -I${CUDA}/include \
    -I${CUDNN}/include \
    -I${VOXELIZE}/include \
    -c \
    -o ${VOXELIZE}/build/voxelize_gpu.o ${VOXELIZE}/src/voxelize.cpp \
    -L${TORCH}/lib \
    -L${CUDA}/lib64 \
    -L${CUDNN}/lib64 \
    -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lcudart \
    -fopenmp

ar rcs ${VOXELIZE}/build/libvoxelize_gpu.a ${VOXELIZE}/build/voxelize_gpu.o
