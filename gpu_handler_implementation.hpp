#ifndef GPU_HANDLER_IMPLEMENTATION_HPP
#define GPU_HANDLER_IMPLEMENTATION_HPP

#include "defines.hpp"

#include <memory>
#include <algorithm>
#include <limits>

#include "cuda.h"
#include "cuda_runtime_api.h"

#include "c10/cuda/CUDAStream.h"

#include "gpu_handler.hpp"
#include "globals.hpp"

// --- Implementation ---

gpu_handler::gpu_handler (const std::string &network_file)
{// {{{
    assert(torch::cuda::is_available());

    Ngpu = torch::cuda::device_count();
    
    #ifndef NDEBUG
    std::fprintf(stderr, "Found %lu GPUs.\n", Ngpu);
    #endif // NDEBUG

    current_gpu = 0;

    // fill the devices
    for (size_t ii=0; ii != Ngpu; ++ii)
        devices.push_back(std::make_shared<c10::Device>(c10::DeviceType::CUDA, ii));

    #ifndef NDEBUG
    std::fprintf(stderr, "gpu_handler : started loading network.\n");
    #endif // NDEBUG

    // load the network onto the CPU
    auto tmp_net = std::make_shared<Net>();
    if (network_file != "NO_FILE")
        torch::load(tmp_net, network_file);

    // we want to evaluate the network
    tmp_net->eval();

    for (auto device : devices)
        networks.push_back(std::dynamic_pointer_cast<Net>(tmp_net->clone(*device)));
    
    #ifndef NDEBUG
    std::fprintf(stderr, "gpu_handler : finished loading network.\n");
    #endif // NDEBUG

    #ifndef NDEBUG
    std::fprintf(stderr, "gpu_handler : started finding the streams.\n");
    #endif // NDEBUG

    // get the streams
    streams.resize(Ngpu);
    for (size_t ii=0; ii != Ngpu; ++ii)
    {
        auto tmp_stream = c10::cuda::getStreamFromPool(false, ii);

        // store the first stream
        streams[ii].push_back(std::make_shared<c10::cuda::CUDAStream>(tmp_stream));

        // loop until we find that we're given the initial stream again
        while ((tmp_stream = c10::cuda::getStreamFromPool(false, ii))
               != *streams[ii][0])
            streams[ii].push_back(std::make_shared<c10::cuda::CUDAStream>(tmp_stream));

        #ifndef NDEBUG
        std::fprintf(stderr, "gpu_handler : Found %lu streams on device #%lu.\n",
                             streams[ii].size(), ii);
        #endif // NDEBUG
    }

    #ifndef NDEBUG
    std::fprintf(stderr, "gpu_handler : finished finding the streams.\n");
    #endif // NDEBUG
}// }}}

inline bool
gpu_handler::get_resource (size_t nbytes,
                           std::shared_ptr<Net> &network,
                           std::shared_ptr<c10::Device> &device,
                           std::shared_ptr<StreamWState> &stream)
{// {{{
    static std::default_random_engine rng;
    static std::uniform_int_distribution<size_t> dist(0);

    // use limit values to indicate that no stream was found
    size_t device_idx = std::numeric_limits<size_t>::max(),
           stream_idx = std::numeric_limits<size_t>::max();

    #ifdef MULTI_ROOT
    #   pragma omp critical (Get_Resource_Critical)
    {
    #endif // MULTI_ROOT

    #ifdef CHECK_FOR_MEM
    // figure out which GPUs have enough memory to hold the tensor we want
    // to push there
    std::vector<bool> gpus_free;
    for (size_t ii=0; ii != Ngpu; ++ii)
    {
        cudaSetDevice((int)ii);
        size_t free_mem, total_mem;
        if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess)
        {
            std::fprintf(stderr, "Unable to read device memory state!\n");
            break;
        }
        gpus_free.push_back(nbytes < 0.8*free_mem);
    }
    #endif // CHECK_FOR_MEM

    #ifdef CHECK_FOR_MEM
    // only continue working if we have found at least one GPU that has
    // free memory
    if (gpus_free.size() == Ngpu
        && std::any_of(gpus_free.cbegin(), gpus_free.cend(),
                       [](bool x){ return x; }))
    {
    #endif // CHECK_FOR_MEM

    #ifndef RANDOM_STREAM
    std::vector<std::pair<size_t,std::vector<size_t>>> idle_stream_indices { Ngpu };

    size_t tmp_current_gpu = current_gpu;

    for (size_t ii=0; ii != Ngpu;
         ++ii, tmp_current_gpu = (tmp_current_gpu+1) % Ngpu)
    {
        // store the GPU index
        idle_stream_indices[ii].first = tmp_current_gpu;

        #ifdef CHECK_FOR_MEM
        if (!gpus_free[tmp_current_gpu])
            continue; // simulate that there are no free streams on this device,
                      // since it is out of memory
        #endif // CHECK_FOR_MEM

        // loop over streams on this GPU
        for (size_t jj=0; jj != streams[tmp_current_gpu].size(); ++jj)
            // add if idle
            if (!streams[ii][jj]->get_busy())
                idle_stream_indices[ii].second.push_back(jj);
    }

    // find the most idle GPU
    auto most_idle = std::max_element(idle_stream_indices.begin(),
                                      idle_stream_indices.end(),
                                      [](const std::pair<size_t,std::vector<size_t>> &a,
                                         const std::pair<size_t,std::vector<size_t>> &b)
                                        { return a.second.size() < b.second.size(); }    );

    // check if all streams are busy (very unlikely case)
    if (!most_idle->second.empty())
    {
        device_idx = most_idle->first;
        stream_idx = most_idle->second[dist(rng) % most_idle->second.size()];
    }

    #else // RANDOM_STREAM

    #ifdef CHECK_FOR_MEM
    do
    {
        device_idx = dist(rng) % Ngpu;
    } while (!gpus_free[device_idx]);
    #else // CHECK_FOR_MEM
    device_idx = dist(rng) % Ngpu;
    #endif // CHECK_FOR_MEM

    stream_idx = dist(rng) % streams[device_idx].size();
    #endif // RANDOM_STREAM

    // block the stream from access -- we do this within a critical region
    // to make sure we don't have any race condition here
    streams[device_idx][stream_idx]->set_busy(true);

    #ifdef CHECK_FOR_MEM
    } // if (any free GPUs, in terms of memory)
    #endif // CHECK_FOR_MEM

    #ifdef MULTI_ROOT
    } // Get_Resource_Critical
    #endif // MULTI_ROOT

    // check if the critical region has not been able to find a resource for us,
    // indicate by return value
    if (device_idx == std::numeric_limits<size_t>::max()
        || stream_idx == std::numeric_limits<size_t>::max())
        return false;
    
    // fill the return values
    network = networks[device_idx];
    device  = devices[device_idx];
    stream  = streams[device_idx][stream_idx];

    // advance the internal index, keeping track of which GPU
    // is most likely to be the next best one
    current_gpu = (device_idx+1) % Ngpu;

    // we have been able to find a resource for the caller
    return true;
}// }}}


#endif // GPU_HANDLER_IMPLEMENTATION_HPP
