#ifndef QUEUES_IMPLEMENTATION_HPP
#define QUEUES_IMPLEMENTATION_HPP

#include "defines.hpp"

#ifndef NDEBUG
#   include <cstdio>
#endif

#include "cuda.h"
#include "cuda_runtime_api.h"

#include "geometry.hpp"
#include "network.hpp"
#include "queues.hpp"
#include "globals.hpp"

// --- Implementation ---

#ifndef WORKERS_MAKE_BATCHES
gpu_queue_item::gpu_queue_item ()
{// {{{
    box_indices.reserve(reserv_size);
    weights.reserve(reserv_size * globals.dim);
    network_inputs.reserve(reserv_size * Net::netw_item_size);
}// }}}

inline void
gpu_queue_item::add (int64_t box_index, std::array<float,3> &cub, float R, const float *weight)
{// {{{
    // add the straightforward things
    box_indices.push_back(box_index);

    #pragma loop_count (1, 2, 3)
    for (int ii=0; ii != globals.dim; ++ii)
        weights.push_back(weight[ii] * std::min(M_4PI_3f32*R*R*R, 1.0F));

    // append the network inputs
    Net::input_normalization(cub, R, network_inputs);
}// }}}

inline bool
gpu_queue_item::is_full ()
{// {{{
    return box_indices.size() >= approx_size;
}// }}}
#endif // WORKERS_MAKE_BATCHES

cpu_queue_item::cpu_queue_item ()
{// {{{
    box_indices.reserve(reserv_size);
    weights.reserve(reserv_size * globals.dim);
    overlaps.reserve(reserv_size);
}// }}}

cpu_queue_item::cpu_queue_item (std::shared_ptr<gpu_batch_queue_item> gpu_result)
#ifdef WORKERS_MAKE_BATCHES
    : box_indices { gpu_result->box_indices },
      weights { gpu_result->weights }
#endif
{// {{{
    #ifndef WORKERS_MAKE_BATCHES
    assert(gpu_result->gpu_inputs.size() > 0);
    #endif // WORKERS_MAKE_BATCHES

    size_t Nitems = gpu_result->current_idx;

    // some debugging checks
    assert(Nitems != 0);
    assert(gpu_result->gpu_tensor.size(0) >= (int64_t)Nitems);

    #ifndef WORKERS_MAKE_BATCHES
    #ifndef NDEBUG
    size_t Nitems1 = 0;
    for (auto x : gpu_result->gpu_inputs)
    {
        assert(x->box_indices.size() * globals.dim == x->weights.size());
        Nitems1 += x->box_indices.size();
    }
    assert(Nitems == Nitems1);
    #endif // NDEBUG
    #endif // WORKERS_MAKE_BATCHES

    #ifndef WORKERS_MAKE_BATCHES
    box_indices.reserve(Nitems);
    weights.reserve(Nitems * globals.dim);
    overlaps.reserve(Nitems);
    for (auto x : gpu_result->gpu_inputs)
    {
        for (size_t ii=0; ii != x->box_indices.size(); ++ii)
            box_indices.push_back(x->box_indices[ii]);
        for (size_t ii=0; ii != x->box_indices.size(); ++ii)
            #pragma loop_count (1, 2, 3)
            for (int64_t dd = 0; dd != globals.dim; ++dd)
                weights.push_back(x->weights[ii*globals.dim+dd]);
    }
    #endif // WORKERS_MAKE_BATCHES

    // get the Tensor back to the CPU
    gpu_result->gpu_tensor = gpu_result->gpu_tensor.to(torch::kCPU);
    gpu_result->gpu_tensor_accessor = gpu_result->gpu_tensor.accessor<float,2>();

    // get the overlaps into this object, with the correct normalization
    for (size_t ii=0; ii != Nitems; ++ii)
        overlaps.push_back(gpu_result->gpu_tensor_accessor[ii][0]);
}// }}}

inline void
cpu_queue_item::add (int64_t box_index, const float *weight, float overlap)
{// {{{
    box_indices.push_back(box_index);

    #pragma loop_count (1, 2, 3)
    for (int ii=0; ii != globals.dim; ++ii)
        weights.push_back(weight[ii]);
    
    overlaps.push_back(overlap);
}// }}}

inline bool
cpu_queue_item::is_full ()
{// {{{
    return box_indices.size() >= approx_size;
}// }}}

inline void
cpu_queue_item::add_to_box ()
{// {{{
    assert(box_indices.size() * globals.dim == weights.size()
           && box_indices.size() == overlaps.size());

    for (size_t ii=0; ii != box_indices.size(); ++ii)
        #pragma loop_count (1, 2, 3)
        for (int64_t dd=0; dd != globals.dim; ++dd)
            #if defined(MULTI_ROOT) && !defined(EXTRA_ROOT_ADD)
            #   pragma omp atomic
            #endif // MULTI_ROOT, EXTRA_ROOT_ADD
            globals.box[globals.dim*box_indices[ii]+dd]
                += weights[ii*globals.dim+dd] * overlaps[ii];
}// }}}

gpu_batch_queue_item::gpu_batch_queue_item () :
    current_idx { 0 },
    gpu_tensor { torch::empty( {batch_size, Net::netw_item_size},
                               device(torch::kCPU)
                                  .pinned_memory(true)
                                  .dtype(torch::kFloat32) ) },
    gpu_tensor_accessor { gpu_tensor.accessor<float,2>() }
{
    #ifdef WORKERS_MAKE_BATCHES
    box_indices.reserve(batch_size);
    weights.reserve(globals.dim * batch_size);
    #endif // WORKERS_MAKE_BATCHES
}

#ifndef WORKERS_MAKE_BATCHES
inline bool
gpu_batch_queue_item::is_full (size_t to_add)
{// {{{
    return current_idx + to_add >= batch_size;
}// }}}
#else // WORKERS_MAKE_BATCHES
inline bool
gpu_batch_queue_item::is_full ()
{
    return box_indices.size() == batch_size;
}
#endif // WORKERS_MAKE_BATCHES

#ifndef WORKERS_MAKE_BATCHES
inline void
gpu_batch_queue_item::add (std::shared_ptr<gpu_queue_item> gpu_input)
{// {{{
    assert(current_idx + gpu_input->box_indices.size() < batch_size);

    // store the pointer to all the data
    gpu_inputs.push_back(gpu_input);

    // write data from this chunk into the batch
    for (size_t ii=0; ii != gpu_input->box_indices.size(); ++ii, ++current_idx)
        for (size_t jj=0; jj != Net::netw_item_size; ++jj)
            gpu_tensor_accessor[current_idx][jj]
                = gpu_input->network_inputs[ii*Net::netw_item_size+jj];
}// }}}
#else // WORKERS_MAKE_BATCHES
inline void
gpu_batch_queue_item::add (int64_t box_index, std::array<float,3> &cub, float R, const float *weight)
{// {{{
    box_indices.push_back(box_index);

    #pragma loop_count (1, 2, 3)
    for (int ii=0; ii != globals.dim; ++ii)
        weights.push_back(weight[ii] * std::min(M_4PI_3f32*R*R*R, 1.0F));

    // write network inputs into tensor
    Net::input_normalization(cub, R, gpu_tensor_accessor[current_idx]);

    ++current_idx;
}// }}}
#endif // WORKERS_MAKE_BATCHES

gpu_process_item::gpu_process_item (std::shared_ptr<gpu_batch_queue_item>  batch_,
                                    std::shared_ptr<c10::Device> device_,
                                    std::shared_ptr<c10::cuda::CUDAStream> stream_,
                                    std::shared_ptr<Net> network_) :
    batch { batch_ },
    device { device_ },
    stream { stream_ },
    network { network_ },
    started { false }
{ }

inline bool
gpu_process_item::is_done ()
{// {{{
    #ifndef STREAMISDONE_TRUE
    return started && stream->query();
    #else // STREAMISDONE_TRUE
    return true;
    #endif // STREAMISDONE_TRUE
}// }}}

inline void
gpu_process_item::compute ()
{// {{{
    assert(batch->gpu_tensor.is_pinned());

    #ifndef NDEBUG
    std::fprintf(stderr, "in gpu_process_item::compute(), trying to compute.\n");
    #endif

    while (true)
    {
        #ifdef CHECK_FOR_MEM
        // figure out whether we have enough memory available
        int gpu_id = device->index();
        cudaSetDevice(gpu_id);
        
        size_t free_mem, total_mem;
        if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess)
        {
            std::fprintf(stderr, "Encountered CUDA Error in gpu_process_item::compute().\n");
            continue;
        }
        
        if (batch->gpu_tensor.nbytes() > 0.8*free_mem)
            continue;
        #endif // CHECK_FOR_MEM

        #ifdef TRY_COMPUTE
        try
        {
        #endif // TRY_COMPUTE
            // establish a Stream context
            at::cuda::CUDAMultiStreamGuard guard (*stream);

            // push the data to the GPU (non-blocking)
            batch->gpu_tensor = batch->gpu_tensor.to(*device, true);

            batch->gpu_tensor = network->forward(batch->gpu_tensor);

            break;
        #ifdef TRY_COMPUTE
        }
        catch (c10::Error &e)
        {
            continue;
        }
        #endif // TRY_COMPUTE
    }
}// }}}


#endif // QUEUES_IMPLEMENTATION_HPP
