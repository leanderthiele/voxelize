#!/bin/sh

export LIBTORCH=${HOME}/libtorch

g++ -std=c++17 -O0 -g3 -Wall -Wextra -Wno-unused-parameter \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    -DTESTS \
    -DMULTI_WORKERS \
    -DMULTI_ROOT \
    -I${LIBTORCH}/include/torch/csrc/api/include/ \
    -I${LIBTORCH}/include \
    voxelize_gpu.cpp \
    -L${LIBTORCH}/lib \
    -ltorch -ltorch_cpu -lc10 -lc10_cuda -lcudart \
    -fopenmp
