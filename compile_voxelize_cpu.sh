#!/bin/sh

export VOXELIZE=.

mkdir -p ${VOXELIZE}/build
mkdir -p ${VOXELIZE}/lib

g++ -std=c++17 -O3 -g3 \
    -Wall -Wextra -Wunused-result -Wno-unused-parameter \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    -DCPU_ONLY \
    -I${VOXELIZE}/include \
    -I${VOXELIZE}/detail \
    -c \
    -o ${VOXELIZE}/build/voxelize_cpu.o ${VOXELIZE}/src/voxelize.cpp \
    -fopenmp

ar rcs ${VOXELIZE}/lib/libvoxelize_cpu.a ${VOXELIZE}/build/voxelize_cpu.o
