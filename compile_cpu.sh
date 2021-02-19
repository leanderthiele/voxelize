#!/bin/sh

g++ -std=c++17 -O3 -g3 \
    -Wall -Wextra -Wunused-result -Wno-unused-parameter \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    -o $1 $1.cpp \
    -lhdf5 -lhdf5_cpp \
    -fopenmp
