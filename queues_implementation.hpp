#ifndef QUEUES_IMPLEMENTATION_HPP
#define QUEUES_IMPLEMENTATION_HPP

#include "queues.hpp"
#include "globals.hpp"

// --- Implementation ---

gpu_queue_item::gpu_queue_item ()
{// {{{
    box_indices.reserve(reserv_size);
    weights.reserve(reserv_size * globals.dim);
    network_inputs.reserve(reserv_size * netw_item_size);
    vol_norm.reserve(reserv_size);
}// }}}

inline void
gpu_queue_item::add (int64_t box_index, std::array<float,3> &cub, float R, const float *weight)
{// {{{
    // add the straightforward things
    box_indices.push_back(box_index);

    #pragma loop_count (1, 2, 3)
    for (int ii=0; ii != globals.dim; ++ii)
        weights.push_back(weight[ii]);

    // add the volume normalization
    vol_norm.push_back(std::min(M_4PI_3f*R*R*R, 1.0F));

    // bring the cube into canonical form
    mod_rotations(cub);

    // now do the network inputs -- TODO : change these depending on specific network setup
    network_inputs.push_back(R);
    for (int ii=0; ii != 3; ++ii)
        network_inputs.push_back(cub[ii]);
    network_inputs.push_back(std::log(R));
    for (int ii=0; ii != 3; ++ii)
        network_inputs.push_back(cub[ii] / R);
}// }}}

inline bool
gpu_queue_item::is_full ()
{// {{{
    return box_indices.size() >= approx_size;
}// }}}

cpu_queue_item::cpu_queue_item ()
{// {{{
    box_indices.reserve(reserv_size);
    weights.reserve(reserv_size * globals.dim);
    overlaps.reserve(reserv_size);
}// }}}

cpu_queue_item::cpu_queue_item (std::shared_ptr<gpu_batch_queue_item> gpu_result)
{// {{{
    assert(gpu_result->gpu_inputs.size() > 0);

    size_t Nitems = gpu_result->current_idx;

    // some debugging checks
    assert(Nitems != 0);
    assert(gpu_result->gpu_tensor.size(0) >= Nitems);

    #ifndef NDEBUG
    size_t Nitems1 = 0;
    for (auto x : gpu_result->gpu_inputs)
    {
        assert(x->box_indices.size() * globals.dim == x->weights.size());
        Nitems1 += x->box_indices.size();
    }
    assert(Nitems == Nitems1);
    #endif // NDEBUG

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

    // get the Tensor back to the CPU
    gpu_result->gpu_tensor.to(torch::kCPU);
    gpu_result->gpu_tensor_accessor = gpu_result->gpu_tensor.accessor<float,2>();

    // get the overlaps into this object
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
            #ifdef MULTI_ROOT
            #   pragma omp atomic
            #endif // MULTI_ROOT
            globals.box[globals.dim*box_indices[ii]+dd]
                += weights[ii*globals.dim+dd] * overlaps[ii];
}// }}}

gpu_batch_queue_item::gpu_batch_queue_item () :
    current_idx { 0 },
    gpu_tensor { torch::empty( {batch_size, netw_item_size},
                               device(torch::kCPU)
                                  .pinned_memory(true)
                                  .dtype(torch::kFloat32) ) },
    gpu_tensor_accessor { gpu_tensor.accessor<float,2>() }
{ }

inline bool
gpu_batch_queue_item::is_full (size_t to_add)
{// {{{
    return current_idx + to_add >= batch_size;
}// }}}

inline void
gpu_batch_queue_item::add (std::shared_ptr<gpu_queue_item> gpu_input)
{// {{{
    assert(current_idx + gpu_input->box_indices.size() < batch_size);

    // store the pointer to all the data
    gpu_inputs.push_back(gpu_input);

    // write data from this chunk into the batch
    for (size_t ii=0; ii != gpu_input->box_indices.size(); ++ii, ++current_idx)
        for (size_t jj=0; jj != netw_item_size; ++jj)
            gpu_tensor_accessor[current_idx][jj]
                = gpu_input->network_inputs[ii*netw_item_size+jj];
}// }}}

inline void
gpu_batch_queue_item::pass_through_net (std::shared_ptr<Net> network)
{// {{{
    gpu_tensor = network->forward(gpu_tensor);
    gpu_tensor_accessor = gpu_tensor.accessor<float,2>();
}// }}}

gpu_process_item::gpu_process_item (std::shared_ptr<gpu_batch_queue_item>  batch_,
                                    std::shared_ptr<c10::Device> device_,
                                    std::shared_ptr<c10::cuda::CUDAStream> stream_,
                                    std::shared_ptr<Net> network_) :
    batch { batch_ },
    device { device_ },
    stream { stream_ },
    network { network_ }
{ }

inline bool
gpu_process_item::is_done ()
{// {{{
    return stream->query();
}// }}}

inline void
gpu_process_item::compute ()
{// {{{
    assert(batch->gpu_tensor.is_pinned());

    // establish a Stream context
    {
        at::cuda::CUDAMultiStreamGuard guard (*stream);

        // push the data to the GPU (non-blocking)
        batch->gpu_tensor.to(*device, true);

        batch->pass_through_net(network);
    }
}// }}}


#endif // QUEUES_IMPLEMENTATION_HPP
