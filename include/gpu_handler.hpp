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

// forward declare Net and StreamWState here so that someone including
// this header file doesn't see the definitions
// (do this outside the namespace so no ambiguities)
class StreamWState;
struct Net;

namespace Voxelize {

class gpu_handler
{// {{{
    // number of available GPU's
    size_t Ngpu;

    #ifndef RANDOM_STREAM
    // index of the gpu we should use next
    size_t current_gpu;
    #endif // RANDOM_STREAM

    std::vector<std::shared_ptr<c10::Device>> devices;
    std::vector<std::vector<std::shared_ptr<StreamWState>>> streams;
    std::vector<std::shared_ptr<Net>> networks;

public :
    float Rmin, Rmax;

    gpu_handler () = default;

    gpu_handler (const std::string &network_file);

    // returns true if it was possible to find an idle stream
    bool get_resource (size_t nbytes,
                       std::shared_ptr<Net> &network,
                       std::shared_ptr<c10::Device> &device,
                       std::shared_ptr<StreamWState> &stream);
};// }}}

} // namespace Voxelize

#endif // GPU_HANDLER_HPP
