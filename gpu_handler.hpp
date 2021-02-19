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
// or whether it is currently busy
class StreamWState
{// {{{
    #ifndef RANDOM_STREAM
    bool is_busy; 
    #endif // RANDOM_STREAM
public :
    c10::cuda::CUDAStream cstream;

    StreamWState (size_t device_idx, bool high_priority=false) :
        #ifndef RANDOM_STREAM
        is_busy { false },
        #endif // RANDOM_STREAM
        cstream { c10::cuda::getStreamFromPool(high_priority, device_idx) }
    { }
    
    bool operator==(const StreamWState &other) const
    {
        // use the overloaded == operator from the CUDAStream wrapper
        return cstream == other.cstream;
    }
    
    bool operator!=(const StreamWState &other) const
    {
        // use the overloaded != operator from the CUDAStream wrapper
        return cstream != other.cstream;
    }

    #ifndef RANDOM_STREAM
    void set_busy (bool new_value)
    {
        // FIXME
        //
        // when using multiple GPUs, this assertion fails!!!
        // (this could also be a CPU race condition since we have more threads now)
        assert(new_value != is_busy);
        is_busy = new_value;
    }
    bool get_busy () const
    {
        return is_busy;
    }
    #endif // RANDOM_STREAM
};// }}}

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
    gpu_handler () = default;

    gpu_handler (const std::string &network_file);

    // returns true if it was possible to find an idle stream
    bool get_resource (size_t nbytes,
                       std::shared_ptr<Net> &network,
                       std::shared_ptr<c10::Device> &device,
                       std::shared_ptr<StreamWState> &stream);
};// }}}

#endif // GPU_HANDLER_HPP
