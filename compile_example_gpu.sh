#!/bin/sh

export VOXELIZE=.

source ${VOXELIZE}/paths.sh

export NETWORK_PATH=./data/network

g++ -std=c++17 -O3 -g3 \
    -Wall -Wextra -Wunused-result -Wno-unused-parameter \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    -UCPU_ONLY \
    -DNETWORK_PATH=\"${NETWORK_PATH}\" \
    -I${TORCH}/include/torch/csrc/api/include/ \
    -I${TORCH}/include \
    -I${CUDA}/include \
    -I${CUDNN}/include \
    -I${VOXELIZE}/include \
    -o example_gpu ${VOXELIZE}/src/example.cpp \
    -L${TORCH}/lib \
    -L${CUDA}/lib64 \
    -L${CUDNN}/lib64 \
    -L${VOXELIZE}/lib \
    -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lcudart \
    -lhdf5 -lhdf5_cpp \
    -lvoxelize_gpu \
    -fopenmp
