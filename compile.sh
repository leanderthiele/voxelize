LIBTORCH = /usr/local/libtorch

g++ -std=c++17
    -D_GLIBCXX_USE_CXX11_ABI=0
    -DTESTS
    -DMULTI_WORKERS
    -I${LIBTORCH}/include/torch/csrc/api/include/
    -I${LIBTORCH}/include
    voxelize.cpp
    -L${LIBTORCH}/lib
    -ltorch -ltorch_cpu -ltorch_gpu -lc10
