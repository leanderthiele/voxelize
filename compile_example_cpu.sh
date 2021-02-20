#!/bin/sh

export VOXELIZE=.
export NETWORK_PATH=./data/network.pt

g++ -std=c++17 -O3 -g3 \
    -Wall -Wextra -Wunused-result -Wno-unused-parameter \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    -DCPU_ONLY \
    -DNETWORK_PATH=${NETWORK_PATH} \
    -I${VOXELIZE}/include \
    -o example_cpu ${VOXELIZE}/src/example.cpp \
    -L${VOXELIZE}/build \
    -lhdf5 -lhdf5_cpp \
    -lvoxelize_cpu \
    -fopenmp
