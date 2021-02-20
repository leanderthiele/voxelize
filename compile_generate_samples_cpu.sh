#!/bin/sh

export VOXELIZE=.
export INPUTS_PATH=${VOXELIZE}/data/inputs.bin
export OUTPUTS_PATH=${VOXELIZE}/data/outputs.bin

mkdir -p ${VOXELIZE}/data

g++ -std=c++17 -O3 -g3 \
    -Wall -Wextra -Wunused-result -Wno-unused-parameter \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    -DINPUTS_PATH=${INPUTS_PATH} \
    -DOUTPUTS_PATH=${OUTPUTS_PATH} \
    -I${VOXELIZE}/include \
    -DCPU_ONLY \
    -o generate_samples_cpu ${VOXELIZE}/src/generate_samples.cpp
