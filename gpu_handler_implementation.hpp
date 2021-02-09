#ifndef GPU_HANDLER_IMPLEMENTATION_HPP
#define GPU_HANDLER_IMPLEMENTATION_HPP

#include "gpu_handler.hpp"
#include "globals.hpp"

// --- Implementation ---

gpu_handler::gpu_handler (const std::string &network_file)
{// {{{
    assert(torch::cuda::is_available());

    Ngpu = torch::cuda::device_count();
    current_gpu = 0;

    // fill the devices
    for (size_t ii=0; ii != Ngpu; ++ii)
        devices.push_back(std::make_shared<c10::Device>(c10::DeviceType::CUDA, ii));

    #ifndef NDEBUG
    std::fprintf(stderr, "gpu_handler : started loading network.\n");
    #endif // NDEBUG

    for (auto device : devices)
    {
        // TODO this current solution requires disk I/O for each device,
        //      but at the moment I can't figure out how the torch::Module::clone thing
        //      works
        networks.emplace_back(new Net);

        // In debugging mode, we give the possibility to not load a network from disk
        #ifndef NDEBUG
        if (network_file != "NO_FILE")
        #endif // NDEBUG
        torch::load(networks.back(), network_file);

        // set to evaluation (as opposed to train) mode
        networks.back()->eval();

        // push to the current device
        networks.back()->to(*device);
    }
    
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
gpu_handler::get_resource (std::shared_ptr<Net> &network,
                           std::shared_ptr<c10::Device> &device,
                           std::shared_ptr<c10::cuda::CUDAStream> &stream)
{// {{{
    static std::default_random_engine rng;
    static std::uniform_int_distribution<size_t> dist(0);

    std::vector<std::pair<size_t,std::vector<size_t>>> idle_stream_indices { Ngpu };

    size_t tmp_current_gpu = current_gpu;

    for (size_t ii=0; ii != Ngpu;
         ++ii, tmp_current_gpu = (tmp_current_gpu+1) % Ngpu)
    {
        // store the GPU index
        idle_stream_indices[ii].first = tmp_current_gpu;

        // loop over streams on this GPU
        for (size_t jj=0; jj != streams[tmp_current_gpu].size(); ++jj)
            // add if idle
            if (streams[ii][jj]->query())
                idle_stream_indices[ii].second.push_back(jj);
    }

    // find the most idle GPU
    auto most_idle = std::max_element(idle_stream_indices.begin(),
                                      idle_stream_indices.end(),
                                      [](const std::pair<size_t,std::vector<size_t>> &a,
                                         const std::pair<size_t,std::vector<size_t>> &b)
                                        { return a.second.size() < b.second.size(); }    );

    // check if all streams are busy (very unlikely case)
    if (most_idle->second.empty())
        return false;

    size_t device_idx = most_idle->first;

    size_t Nstreams = most_idle->second.size();
    size_t stream_idx;
    // loop until we find an idle stream (should be on the first iteration)
    while (true)
    {
        size_t r = dist(rng) % Nstreams;
        if (streams[device_idx][most_idle->second[r]]->query())
        {
            stream_idx = most_idle->second[r];
            break;
        }
    }
    
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
