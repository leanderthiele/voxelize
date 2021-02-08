#ifndef GPU_HANDLER_HPP
#define GPU_HANDLER_HPP

#include <vector>
#include <memory>
#include <utility>

#include <torch/torch.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAStream.h>

class gpu_handler
{
    // number of available GPU's
    size_t Ngpu;

    // index of the gpu we should use next
    size_t current_gpu;

    std::vector<std::shared_ptr<c10::Device>> devices;
    std::vector<std::vector<std::shared_ptr<c10::cuda::CUDAStream>>> streams;

public :
    gpu_handler ();

    // returns true if it was possible to find an idle stream
    bool get_resource (int &device_idx,
                       std::shared_ptr<c10::Device> &device,
                       std::shared_ptr<c10::cuda::CUDAStream> &stream);
};

// --- Implementation ---

gpu_handler::gpu_handler ()
{
    assert(torch::cuda::is_available());

    Ngpu = torch::cuda::device_count();
    current_gpu = 0;

    // fill the devices
    for (size_t ii=0; ii != Ngpu; ++ii)
        devices.push_back(std::make_shared<c10::Device>(c10::DeviceType::CUDA, ii));
    
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
    }
}

inline bool
gpu_handler::get_resource (int &device_idx,
                           std::shared_ptr<c10::Device> &device,
                           std::shared_ptr<c10::cuda::CUDAStream> &stream)
{
    // keep track of where we started our search
    size_t gpu_search_start = current_gpu;

    // loop until we find a suitable one
    do
    {

        for (auto stream_ : streams[current_gpu])
            // check if stream is busy
            if (stream_->query())
            {
                device_idx = current_gpu;
                device = devices[current_gpu];
                stream = stream_;
                current_gpu = (current_gpu+1) % Ngpu;
                return true;
            }
        
        // go to the next one
        current_gpu = (current_gpu+1) % Ngpu;

    } while (current_gpu != gpu_search_start);

    // search not successful, every single stream is busy
    return false;
}

#endif // GPU_HANDLER_HPP
