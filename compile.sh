#!/bin/sh

export TORCH=${HOME}/pytorch-install
export CUDNN=/usr/local/cudnn/cuda-11.0/8.0.2

g++ -std=c++17 -O3 -g3 \
    -Wall -Wextra -Wunused-result -Wno-unused-parameter \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    -DTESTS \
    -DMULTI_WORKERS \
    -DWORKERS_MAKE_BATCHES \
    -DMULTI_ROOT \
    -DCOUNT \
    -DNDEBUG \
    -DRANDOM_STREAM \
    -DEXTRA_ADD_ROOT \
    -I${TORCH}/include/torch/csrc/api/include/ \
    -I${TORCH}/include \
    -I${CUDNN}/include \
    -o $1 $1.cpp \
    -L${TORCH}/lib \
    -L${CUDNN}/lib64 \
    -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lcudart \
    -lhdf5 -lhdf5_cpp \
    -fopenmp
