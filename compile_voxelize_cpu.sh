#!/bin/sh

export VOXELIZE=.

mkdir -p build

g++ -std=c++17 -O3 -g3 \
    -Wall -Wextra -Wunused-result -Wno-unused-parameter \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    -DCPU_ONLY \
    -o ${VOXELIZE}/build/voxelize_cpu.o ${VOXELIZE}/src/voxelize.cpp \
    -fopenmp

ar rcs ${VOXELIZE}/build/libvoxelize_cpu.a ${VOXELIZE}/src/build/voxelize_cpu.o
