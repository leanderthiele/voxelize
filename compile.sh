g++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -I/usr/local/libtorch/include/torch/csrc/api/include/ -I/usr/local/libtorch/include main.cpp -L/usr/local/libtorch/lib -ltorch -ltorch_cpu -lc10
