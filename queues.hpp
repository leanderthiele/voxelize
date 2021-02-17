#ifndef QUEUES_HPP
#define QUEUES_HPP

#include <cassert>
#include <array>
#include <vector>
#include <queue>
#include <forward_list>
#include <memory>
#include <cmath>
#include <algorithm>

#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAMultiStreamGuard.h>

#include "geometry.hpp"
#include "network.hpp"

// The different queues (items forward declared, implementation later in this file)
struct cpu_queue_item;
struct gpu_queue_item;
struct gpu_batch_queue_item;
struct gpu_process_item;

// lives on shared memory, all workers and root insert
// Working along the queue means adding items to the output box.
typedef std::queue<std::shared_ptr<cpu_queue_item>>          cpu_queue_t;

#ifndef WORKERS_MAKE_BATCHES
// lives on shared memory, all workers insert
// Working along the queue means assembling gpu_batch_queue_items
typedef std::queue<std::shared_ptr<gpu_queue_item>>          gpu_queue_t;
#endif // WORKERS_MAKE_BATCHES

// lives on root memory, root inserts
// Working along the queue means constructing gpu_process_items
typedef std::queue<std::shared_ptr<gpu_batch_queue_item>>    gpu_batch_queue_t;

// lives on root memory, root inserts
// Working along the list means constructing cpu_queue_items
typedef std::forward_list<std::shared_ptr<gpu_process_item>> gpu_process_list_t;

#ifndef WORKERS_MAKE_BATCHES
struct gpu_queue_item
{// {{{
    // this is the approximate size we're aiming for
    // (some will be a bit larger, others a bit smaller)
    static constexpr size_t approx_size = 4000;

    // this is the size we'll allocate
    // Hopefully we'll rarely need re-allocs
    static constexpr size_t reserv_size = 4096;

    std::vector<int64_t> box_indices;
    std::vector<float>   weights;
    std::vector<float>   network_inputs;
    std::vector<float>   vol_norm; // this is the volume normalization, used in the earlier implementation
                                   // TODO -- when everything is working, we can merge this with the weights
                                   //         vector, saving the memory

    // constructor
    gpu_queue_item ();

    // add one item requiring computation on the gpu
    void add (int64_t box_index, std::array<float,3> &cub, float R, const float *weight);

    bool is_full ();
};// }}}
#endif // WORKERS_MAKE_BATCHES

// Note : this struct is a one-use object, after calling compute the results
//        can be retrieved but in the current implementation it has to be
//        discarded afterwards
struct gpu_batch_queue_item
{// {{{
    static constexpr size_t batch_size = 16*32768;

    size_t current_idx;
    #ifndef WORKERS_MAKE_BATCHES
    std::vector<std::shared_ptr<gpu_queue_item>> gpu_inputs;
    #else // WORKERS_MAKE_BATCHES
    std::vector<int64_t> box_indices;
    std::vector<float>   weights;
    std::vector<float>   vol_norm; // TODO -- when everything is working,
                                   //         we can merge this with the weights vector
    #endif // WORKERS_MAKE_BATCHES

    torch::Tensor gpu_tensor;
    torch::TensorAccessor<float,2> gpu_tensor_accessor;

    // constructor
    gpu_batch_queue_item ();

    #ifndef WORKERS_MAKE_BATCHES
    bool is_full (size_t to_add);
    #else // WORKERS_MAKE_BATCHES
    bool is_full ();
    #endif // WORKERS_MAKE_BATCHES

    #ifndef WORKERS_MAKE_BATCHES
    void add (std::shared_ptr<gpu_queue_item> gpu_input);
    #else // WORKERS_MAKE_BATCHES
    void add (int64_t box_index, std::array<float,3> &cub, float R, const float *weight);
    #endif // WORKERS_MAKE_BATCHES
};// }}}

struct cpu_queue_item
{// {{{
    // these are for construction with trivial results in the worker threads
    static constexpr size_t approx_size = 4000;
    static constexpr size_t reserv_size = 4096;

    std::vector<int64_t> box_indices;
    std::vector<float>   weights;
    std::vector<float>   overlaps;

    // default constructor (called on worker threads)
    cpu_queue_item ();

    // specialized constructor (called on root thread with network result)
    cpu_queue_item (std::shared_ptr<gpu_batch_queue_item> gpu_result);

    void add (int64_t box_index, const float *weight, float overlap);

    bool is_full ();

    // the main functionality of this class: transfer the data into the main box
    void add_to_box ();
};// }}}

struct gpu_process_item
{// {{{
    std::shared_ptr<gpu_batch_queue_item> batch;

    std::shared_ptr<c10::Device> device;
    std::shared_ptr<c10::cuda::CUDAStream> stream;
    std::shared_ptr<Net> network;

    gpu_process_item (std::shared_ptr<gpu_batch_queue_item>  batch_,
                      std::shared_ptr<c10::Device> device_,
                      std::shared_ptr<c10::cuda::CUDAStream> stream_,
                      std::shared_ptr<Net> network_);
    
    void compute ();

    bool is_done ();
};// }}}


#endif // QUEUES_HPP
