#!/bin/sh

export VOXELIZE=.

g++ -std=c++17 -O3 -g3 \
    -Wall -Wextra -Wunused-result -Wno-unused-parameter \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    -DCPU_ONLY \
    -I${VOXELIZE}/include \
    -o example_cpu ${VOXELIZE}/src/example.cpp \
    -L${VOXELIZE}/lib \
    -lhdf5 -lhdf5_cpp \
    -lvoxelize_cpu \
    -fopenmp
