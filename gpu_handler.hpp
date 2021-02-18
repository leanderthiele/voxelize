#ifndef GPU_HANDLER_HPP
#define GPU_HANDLER_HPP

#include "defines.hpp"

#include <vector>
#include <memory>
#include <utility>
#include <algorithm>
#include <random>
#include <string>

#ifndef NDEBUG
#   include <cstdio>
#endif // NDEBUG

#include <torch/torch.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAStream.h>

#include "network.hpp"

class gpu_handler
{// {{{
    // number of available GPU's
    size_t Ngpu;

    // index of the gpu we should use next
    size_t current_gpu;

    std::vector<std::shared_ptr<c10::Device>> devices;
    std::vector<std::vector<std::shared_ptr<c10::cuda::CUDAStream>>> streams;
    std::vector<std::shared_ptr<Net>> networks;

public :
    gpu_handler () = default;

    gpu_handler (const std::string &network_file);

    // returns true if it was possible to find an idle stream
    bool get_resource (std::shared_ptr<Net> &network,
                       std::shared_ptr<c10::Device> &device,
                       std::shared_ptr<c10::cuda::CUDAStream> &stream);
};// }}}

#endif // GPU_HANDLER_HPP
