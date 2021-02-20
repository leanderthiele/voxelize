#ifndef QUEUES_IMPLEMENTATION_HPP
#define QUEUES_IMPLEMENTATION_HPP

#include "defines.hpp"

#include <cstdio>
#include <exception>

#ifndef CPU_ONLY
#   include "cuda.h"
#   include "cuda_runtime_api.h"
#endif // CPU_ONLY

#ifdef CPU_ONLY
#   include "voxelize_cpu.hpp"
#else // CPU_ONLY
#   include "voxelize_gpu.hpp"
#endif // CPU_ONLY

#include "geometry.hpp"
#include "queues.hpp"
#include "globals.hpp"

#ifndef CPU_ONLY
#   include "network.hpp"
#   include "gpu_handler.hpp"
#   include "gpu_handler_implementation.hpp"
#endif // CPU_ONLY

using namespace Voxelize;

// --- Implementation ---

// simple Macro to choose a templated version of a (void) function (based on globals.dim)
#define CHOOSE_TEMPL(fct, ...)                          \
    do                                                  \
    {                                                   \
        switch (globals.dim)                            \
        {                                               \
            case (1) : fct<1>(__VA_ARGS__); break;      \
            case (2) : fct<2>(__VA_ARGS__); break;      \
            case (3) : fct<3>(__VA_ARGS__); break;      \
            default  : std::fprintf(stderr,             \
                       "Dimension %ld not implemented", \
                       globals.dim); std::terminate();  \
        }                                               \
    } while (0)

#ifndef CPU_ONLY
#ifndef WORKERS_MAKE_BATCHES
gpu_queue_item::gpu_queue_item ()
{// {{{
    box_indices.reserve(reserv_size);
    weights.reserve(reserv_size * globals.dim);
    network_inputs.reserve(reserv_size * Net::netw_item_size);
}// }}}
#endif // WORKERS_MAKE_BATCHES
#endif // CPU_ONLY

#ifndef CPU_ONLY
#ifndef WORKERS_MAKE_BATCHES
template<int DIM>
inline void
gpu_queue_item::add (int64_t box_index, std::array<float,3> &cub, float R, const float *weight)
{// {{{
    // add the straightforward things
    box_indices.push_back(box_index);

    for (int ii=0; ii != DIM; ++ii)
        weights.push_back(weight[ii] * std::min(M_4PI_3f32*R*R*R, 1.0F));

    // append the network inputs
    Net::input_normalization(cub, R, network_inputs);
}// }}}

inline void
gpu_queue_item::add (int64_t box_index, std::array<float,3> &cub, float R, const float *weight)
{// {{{
    CHOOSE_TEMPL(add, box_index, cub, R, weight);
}// }}}
#endif // WORKERS_MAKE_BATCHES
#endif // CPU_ONLY

#ifndef CPU_ONLY
#ifndef WORKERS_MAKE_BATCHES
inline bool
gpu_queue_item::is_full ()
{// {{{
    return box_indices.size() >= approx_size;
}// }}}
#endif // WORKERS_MAKE_BATCHES
#endif // CPU_ONLY

cpu_queue_item::cpu_queue_item ()
{// {{{
    box_indices.reserve(batch_size);
    weights.reserve(batch_size * globals.dim);
    overlaps.reserve(batch_size);
}// }}}

#ifndef CPU_ONLY
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
        for (size_t ii=0; ii != x->box_indices.size() * globals.dim; ++ii)
                weights.push_back(x->weights[ii]);
    }
    #endif // WORKERS_MAKE_BATCHES

    // check if we really have a CPU tensor
    assert(gpu_result->gpu_tensor.device() == torch::kCPU);

    // update the tensor accessor
    gpu_result->gpu_tensor_accessor = gpu_result->gpu_tensor.accessor<float,2>();

    // get the overlaps into this object, with the correct normalization
    for (size_t ii=0; ii != Nitems; ++ii)
    {
        #ifndef NDEBUG
        if (gpu_result->gpu_tensor_accessor[ii][0] < 0.0F)
            std::fprintf(stderr, "Negative network output : %.4e\n",
                                 gpu_result->gpu_tensor_accessor[ii][0]);
        #endif
        assert(gpu_result->gpu_tensor_accessor[ii][0] >= 0.0F);
        overlaps.push_back(gpu_result->gpu_tensor_accessor[ii][0]);
    }
}// }}}
#endif // CPU_ONLY

template<int DIM>
inline void
cpu_queue_item::add (int64_t box_index, const float *weight, float overlap)
{// {{{
    box_indices.push_back(box_index);

    for (int ii=0; ii != DIM; ++ii)
        weights.push_back(weight[ii]);
    
    assert(overlap >= 0.0F);
    overlaps.push_back(overlap);
}// }}}

inline void
cpu_queue_item::add (int64_t box_index, const float *weight, float overlap)
{// {{{
    CHOOSE_TEMPL(add, box_index, weight, overlap);
}// }}}

inline bool
cpu_queue_item::is_full ()
{// {{{
    return box_indices.size() >= batch_size;
}// }}}

template<int DIM>
inline void
cpu_queue_item::add_to_box ()
{// {{{
    assert(box_indices.size() * globals.dim == weights.size()
           && box_indices.size() == overlaps.size());

    for (size_t ii=0; ii != box_indices.size(); ++ii)
        for (int64_t dd=0; dd != DIM; ++dd)
        {
            assert(overlaps[ii] >= 0.0F);

            #if defined(MULTI_ROOT) && !defined(EXTRA_ROOT_ADD) && !defined(CPU_ONLY)
            #   pragma omp atomic
            #endif // MULTI_ROOT, EXTRA_ROOT_ADD, CPU_ONLY
            globals.box[DIM*box_indices[ii]+dd]
                += weights[ii*DIM+dd] * overlaps[ii];
        }
}// }}}

inline void
cpu_queue_item::add_to_box ()
{// {{{
    CHOOSE_TEMPL(add_to_box);
}// }}}

#ifndef CPU_ONLY
gpu_batch_queue_item::gpu_batch_queue_item () :
    current_idx { 0 },
    gpu_tensor { torch::empty( {batch_size, Net::netw_item_size},
                               device(torch::kCPU)
                                  .pinned_memory(true)
                                  .dtype(torch::kFloat32)
                                  .requires_grad(false) ) },
    gpu_tensor_accessor { gpu_tensor.accessor<float,2>() }
{// {{{
    #ifdef WORKERS_MAKE_BATCHES
    box_indices.reserve(batch_size);
    weights.reserve(globals.dim * batch_size);
    #endif // WORKERS_MAKE_BATCHES
}// }}}
#endif // CPU_ONLY

#ifndef CPU_ONLY
#ifndef WORKERS_MAKE_BATCHES
inline bool
gpu_batch_queue_item::is_full (size_t to_add)
{// {{{
    return current_idx + to_add >= batch_size;
}// }}}
#else // WORKERS_MAKE_BATCHES
inline bool
gpu_batch_queue_item::is_full ()
{// {{{
    return box_indices.size() == batch_size;
}// }}}
#endif // WORKERS_MAKE_BATCHES
#endif // CPU_ONLY

#ifndef CPU_ONLY
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
template<int DIM>
inline void
gpu_batch_queue_item::add (int64_t box_index, std::array<float,3> &cub, float R, const float *weight)
{// {{{
    box_indices.push_back(box_index);

    for (int ii=0; ii != DIM; ++ii)
        weights.push_back(weight[ii] * std::min(M_4PI_3f32*R*R*R, 1.0F));

    // write network inputs into tensor
    Net::input_normalization(cub, R, gpu_tensor_accessor[current_idx]);

    ++current_idx;
}// }}}

inline void
gpu_batch_queue_item::add (int64_t box_index, std::array<float,3> &cub, float R, const float *weight)
{// {{{
    CHOOSE_TEMPL(add, box_index, cub, R, weight);
}// }}}
#endif // WORKERS_MAKE_BATCHES
#endif // CPU_ONLY

#ifndef CPU_ONLY
gpu_process_item::gpu_process_item (std::shared_ptr<gpu_batch_queue_item>  batch_,
                                    std::shared_ptr<c10::Device> device_,
                                    std::shared_ptr<StreamWState> stream_,
                                    std::shared_ptr<Net> network_) :
    batch { batch_ },
    device { device_ },
    stream { stream_ },
    network { network_ },
    finished_event { }
{// {{{
    // perform the computation (asynchronously)
    compute();
}// }}}
#endif // CPU_ONLY

#ifndef CPU_ONLY
inline void
gpu_process_item::release_resources () const
{// {{{
    assert(is_done());
    #ifndef RANDOM_STREAM
    stream->set_busy(false);
    #endif // RANDOM_STREAM
}// }}}
#endif // CPU_ONLY

#ifndef CPU_ONLY
inline bool
gpu_process_item::is_done () const
{// {{{
    return finished_event.isCreated() && finished_event.query();
}// }}}
#endif // CPU_ONLY

#ifndef CPU_ONLY
inline void
gpu_process_item::compute ()
{// {{{
    assert(batch->gpu_tensor.is_pinned());

    {
        // establish a Stream context
        c10::cuda::CUDAStreamGuard guard (stream->cstream);

        // push the data to the GPU (non-blocking)
        batch->gpu_tensor = batch->gpu_tensor.to(*device, /*non_blocking=*/true);

        // pass the data through the network
        batch->gpu_tensor = network->forward(batch->gpu_tensor);

        // push the data to the CPU (non-blocking)
        batch->gpu_tensor = batch->gpu_tensor.to(torch::kCPU, /*non_blocking=*/true);

        // record that we are done here
        finished_event.record();
    }
}// }}}
#endif // CPU_ONLY

#undef CHOOSE_TEMPL

#endif // QUEUES_IMPLEMENTATION_HPP
