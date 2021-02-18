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

// a wrapper around a CUDA Stream that captures
// whether the stream can be used for computations
// of whether it is currently busy
class StreamWState : public c10::cuda::CUDAStream
{// {{{
    bool is_busy = false; 
public :
    void set_busy (bool new_value)
    {
        assert(new_value != is_busy);
        is_busy = new_value;
    }
    bool get_busy () const
    {
        return is_busy;
    }
};// }}}

class gpu_handler
{// {{{
    // number of available GPU's
    size_t Ngpu;

    // index of the gpu we should use next
    size_t current_gpu;

    std::vector<std::shared_ptr<c10::Device>> devices;
    std::vector<std::vector<std::shared_ptr<StreamWState>>> streams;
    std::vector<std::shared_ptr<Net>> networks;

public :
    gpu_handler () = default;

    gpu_handler (const std::string &network_file);

    // returns true if it was possible to find an idle stream
    bool get_resource (size_t nbytes,
                       std::shared_ptr<Net> &network,
                       std::shared_ptr<c10::Device> &device,
                       std::shared_ptr<StreamWState> &stream);
};// }}}

#endif // GPU_HANDLER_HPP
