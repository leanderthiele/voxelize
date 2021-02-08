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

#include "geometry.hpp"

// The different queues (items forward declared, implementation later in this file)
struct cpu_queue_item;
struct gpu_queue_item;
struct gpu_batch_queue_item;
struct gpu_process_item;

// lives on shared memory, all workers and root insert
// Working along the queue means adding items to the output box.
typedef std::queue<std::shared_ptr<cpu_queue_item>>          cpu_queue_t;

// lives on shared memory, all workers insert
// Working along the queue means assembling gpu_batch_queue_items
typedef std::queue<std::shared_ptr<gpu_queue_item>>          gpu_queue_t;

// lives on root memory, root inserts
// Working along the queue means constructing gpu_process_items
typedef std::queue<std::shared_ptr<gpu_batch_queue_item>>    gpu_batch_queue_t;

// lives on root memory, root inserts
// Working along the list means constructing cpu_queue_items
typedef std::forward_list<std::shared_ptr<gpu_process_item>> gpu_process_list_t;


struct gpu_queue_item
{// {{{
    // this is the approximate size we're aiming for
    // (some will be a bit larger, others a bit smaller)
    static constexpr size_t approx_size = 100;

    // this is the size we'll allocate
    // Hopefully we'll rarely need re-allocs
    static constexpr size_t reserv_size = 128;

    // this is the network input size per item, later we'll probably define it somewhere else
    static constexpr size_t netw_item_size = 8;

    int64_t dim;
    std::vector<int64_t> box_indices;
    std::vector<float>   weights;
    std::vector<float>   network_inputs;
    std::vector<float>   vol_norm; // this is the volume normalization, used in the earlier implementation
                                   // TODO -- when everything is working, we can merge this with the weights
                                   //         vector, saving the memory

    // constructor
    gpu_queue_item (int64_t dim_);

    // add one item requiring computation on the gpu
    void add (int64_t box_index, std::array<float,3> &cub, float R, const float *weight);

    bool is_full ();
};// }}}

// Note : this struct is a one-use object, after calling compute the results
//        can be retrieved but in the current implementation it has to be
//        discarded afterwards
struct gpu_batch_queue_item
{// {{{
    static constexpr size_t batch_size = 4096;
    static constexpr size_t netw_item_size = 8;

    size_t current_idx;
    std::vector<std::shared_ptr<gpu_queue_item>> gpu_inputs;
    torch::Tensor gpu_tensor;
    torch::TensorAccessor<float,2> gpu_tensor_accessor;

    // constructor
    gpu_batch_queue_item ();

    bool is_full (size_t to_add);

    void add (std::shared_ptr<gpu_queue_item> gpu_input);

    void compute ();
};// }}}

struct cpu_queue_item
{// {{{
    // these are for construction with trivial results in the worker threads
    static constexpr size_t approx_size = 100;
    static constexpr size_t reserv_size = 128;

    int64_t dim;
    std::vector<int64_t> box_indices;
    std::vector<float>   weights;
    std::vector<float>   overlaps;

    // default constructor (called on worker threads)
    cpu_queue_item (int64_t dim_);

    // specialized constructor (called on root thread with network result)
    cpu_queue_item (const gpu_batch_queue_item &gpu_result);

    void add (int64_t box_index, const float *weight, float overlap);

    bool is_full ();

    // the main functionality of this class: transfer the data into the main box
    void add_to_box (float *box);
};// }}}

struct gpu_process_item
{// {{{
    std::shared_ptr<gpu_batch_queue_item>  batch;
    std::shared_ptr<c10::cuda::CUDAStream> stream;

    gpu_process_item (std::shared_ptr<gpu_batch_queue_item>  batch_,
                      std::shared_ptr<c10::cuda::CUDAStream> stream_);

    bool is_done ();
};// }}}

// --- Implementation ---

gpu_queue_item::gpu_queue_item (int64_t dim_) : dim(dim_)
{// {{{
    box_indices.reserve(reserv_size);
    weights.reserve(reserv_size * dim);
    network_inputs.reserve(reserv_size * netw_item_size);
    vol_norm.reserve(reserv_size);
}// }}}

inline void
gpu_queue_item::add (int64_t box_index, std::array<float,3> &cub, float R, const float *weight)
{// {{{
    // add the straightforward things
    box_indices.push_back(box_index);

    #pragma loop_count (1, 2, 3)
    for (int ii=0; ii != dim; ++ii)
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

cpu_queue_item::cpu_queue_item (int64_t dim_) : dim(dim_)
{// {{{
    box_indices.reserve(reserv_size);
    weights.reserve(reserv_size * dim);
    overlaps.reserve(reserv_size);
}// }}}

cpu_queue_item::cpu_queue_item (const gpu_batch_queue_item &gpu_result)
{// {{{
    assert(gpu_result.gpu_inputs.size() > 0);

    dim = gpu_result.gpu_inputs[0]->dim;

    size_t Nitems = gpu_result.current_idx;

    // some debugging checks
    assert(Nitems != 0);
    assert(gpu_result.gpu_tensor.size(0) >= Nitems);
    #ifndef NDEBUG
    size_t Nitems1 = 0;
    for (auto x : gpu_result.gpu_inputs)
    {
        assert(x->dim == dim);
        assert(x->box_indices.size() * dim == x->weights.size());
        Nitems1 += x->box_indices.size();
    }
    assert(Nitems == Nitems1);
    #endif

    box_indices.reserve(Nitems);
    weights.reserve(Nitems * dim);
    overlaps.reserve(Nitems);

    for (auto x : gpu_result.gpu_inputs)
    {
        for (size_t ii=0; ii != x->box_indices.size(); ++ii)
            box_indices.push_back(x->box_indices[ii]);
        for (size_t ii=0; ii != x->box_indices.size(); ++ii)
            #pragma loop_count (1, 2, 3)
            for (int64_t dd = 0; dd != dim; ++dd)
                weights.push_back(x->weights[ii*dim+dd]);
    }    

    for (size_t ii=0; ii != Nitems; ++ii)
        overlaps.push_back(gpu_result.gpu_tensor_accessor[ii][0]);
}// }}}

inline void
cpu_queue_item::add (int64_t box_index, const float *weight, float overlap)
{// {{{
    box_indices.push_back(box_index);

    #pragma loop_count (1, 2, 3)
    for (int ii=0; ii != dim; ++ii)
        weights.push_back(weight[ii]);
    
    overlaps.push_back(overlap);
}// }}}

inline bool
cpu_queue_item::is_full ()
{// {{{
    return box_indices.size() >= approx_size;
}// }}}

inline void
cpu_queue_item::add_to_box(float *box)
{// {{{
    assert(box_indices.size() == weights.size()
           && box_indices.size() == overlaps.size());

    for (size_t ii=0; ii != box_indices.size(); ++ii)
        #pragma loop_count (1, 2, 3)
        for (int64_t dd=0; dd != dim; ++dd)
            box[dim*box_indices[ii]+dd] += weights[ii*dim+dd] * overlaps[ii];
}// }}}

gpu_batch_queue_item::gpu_batch_queue_item () :
    current_idx { 0 },
    gpu_tensor { torch::empty( {batch_size, netw_item_size}, torch::kFloat32 ) },
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
gpu_batch_queue_item::compute ()
{// {{{
    // gpu_tensor.to( GPU ) TODO
    // gpu_tensor = NETWORK ( gpu_tensor ) TODO
    // gpu_tensor.to( CPU ) TODO

    gpu_tensor_accessor = gpu_tensor.accessor<float,2>();
}// }}}

gpu_process_item::gpu_process_item (std::shared_ptr<gpu_batch_queue_item>  batch_,
                                    std::shared_ptr<c10::cuda::CUDAStream> stream_) :
    batch { batch_ },
    stream { stream_ }
{ }

inline bool
gpu_process_item::is_done ()
{
    return stream->query();
}


#endif // QUEUES_HPP
