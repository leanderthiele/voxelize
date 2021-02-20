#ifndef ROOT_HPP
#define ROOT_HPP

#include "defines.hpp"

#include <memory>
#include <omp.h>

#ifndef NDEBUG
#   include <cstdio>
#endif

#ifndef CPU_ONLY
#   include <torch/torch.h>
#   include <c10/cuda/CUDAStream.h>
#endif // CPU_ONLY

#ifdef CPU_ONLY
#   include "voxelize_cpu.hpp"
#else // CPU_ONLY
#   include "voxelize_gpu.hpp"
#endif // CPU_ONLY

#include "queues.hpp"
#include "queues_implementation.hpp"
#include "globals.hpp"

#ifndef CPU_ONLY
#   include "network.hpp"
#   include "gpu_handler.hpp"
#   include "gpu_handler_implementation.hpp"
#endif // CPU_ONLY

using namespace Voxelize;

extern Globals globals;

#ifndef CPU_ONLY
#ifndef WORKERS_MAKE_BATCHES
static inline void
check_gpu_queue (std::shared_ptr<gpu_batch_queue_item> &gpu_batch_queue_item_ptr,
                 bool can_finish)
{// {{{
    std::shared_ptr<gpu_queue_item> gpu_queue_item_ptr;

    #pragma omp critical (GPU_Queue_Critical)
    if (!globals.gpu_queue.empty())
    {
        assert(!can_finish);

        gpu_queue_item_ptr = globals.gpu_queue.front();
        globals.gpu_queue.pop();
    }

    if (gpu_queue_item_ptr || can_finish)
    {
        // check if we have a sufficiently full batch to push it to the GPU queue,
        // or this is the last batch (and it has data)
        // Note the order of the ||, since in the second component we need to use
        // the gpu_queue_item_ptr which is uninitialized if can_finish==true.
        if ((can_finish && gpu_batch_queue_item_ptr->current_idx) ||
            gpu_batch_queue_item_ptr->is_full(gpu_queue_item_ptr->box_indices.size()))
        {
            #ifdef MULTI_ROOT
            #   pragma omp critical (GPU_Batch_Queue_Critical)
            #endif // MULTI_ROOT
            globals.gpu_batch_queue.push(gpu_batch_queue_item_ptr);
            
            // now we can reset this pointer to a pristine batch
            gpu_batch_queue_item_ptr.reset(new gpu_batch_queue_item);
        }

        if (!can_finish)
            // append this gpu_item to the batch
            gpu_batch_queue_item_ptr->add(gpu_queue_item_ptr);
    }
}// }}}
#endif // WORKERS_MAKE_BATCHES
#endif // CPU_ONLY

#ifndef CPU_ONLY
static inline void
#ifdef COUNT
check_gpu_batch_queue (uint64_t &processed_batches)
#else // COUNT
check_gpu_batch_queue ()
#endif // COUNT
{// {{{
    std::shared_ptr<gpu_batch_queue_item> gpu_batch_queue_item_ptr;

    #ifdef MULTI_ROOT
    #   pragma omp critical (GPU_Batch_Queue_Critical)
    #endif // MULTI_ROOT
    if (!globals.gpu_batch_queue.empty())
    {
        gpu_batch_queue_item_ptr = globals.gpu_batch_queue.front();
        globals.gpu_batch_queue.pop();
    }

    if (gpu_batch_queue_item_ptr)
    {
        #ifdef COUNT
        ++processed_batches;
        #endif // COUNT

        std::shared_ptr<Net> network_ptr;
        std::shared_ptr<c10::Device> device_ptr;
        std::shared_ptr<StreamWState> stream_ptr;

        // figure out how much memory we'd need
        size_t nbytes = gpu_batch_queue_item_ptr->gpu_tensor.nbytes();

        // find a stream for this calculation
        if (!globals.gpu->get_resource(nbytes, network_ptr, device_ptr, stream_ptr))
        {
            // append this batch to the GPU -- we do a preliminary pop in the previous
            // critical section since we expect that in the majority of cases
            // we'll be able to obtain a resource
            #ifdef MULTI_ROOT
            #   pragma omp critical (GPU_Batch_Queue_Critical)
            #endif // MULTI_ROOT
            globals.gpu_batch_queue.push(gpu_batch_queue_item_ptr);

            // continue execution if we cannot find a resource
            return;
        }

        // if a stream has been found, start a GPU process,
        // which is implicitly done by the gpu_process_item constructor.
        // We append the new gpu_process_item to the end of the list,
        // so that looping through the list in check_gpu_process_list()
        // gives a finished process early.
        #ifdef MULTI_ROOT
        #   pragma omp critical (GPU_Process_List_Critical)
        #endif // MULTI_ROOT
        globals.gpu_process_list
            .emplace_back(new gpu_process_item(gpu_batch_queue_item_ptr,
                                               device_ptr,
                                               stream_ptr,
                                               network_ptr));
    }
}// }}}
#endif // CPU_ONLY

#ifndef CPU_ONLY
static inline void
check_gpu_process_list ()
{// {{{
    std::shared_ptr<gpu_process_item> gpu_process_item_ptr;
    std::shared_ptr<cpu_queue_item> cpu_queue_item_ptr;

    // check if we have finished gpu computations whose results we can push to the CPU queue
    #ifdef MULTI_ROOT
    #   pragma omp critical (GPU_Process_List_Critical)
    #endif // MULTI_ROOT
    for (auto x : globals.gpu_process_list)
        if (x->is_done())
        {
            gpu_process_item_ptr = x;
            globals.gpu_process_list.remove(x);
            break;
        }

    // if we found finished stuff in the GPU process list, do some work
    if (gpu_process_item_ptr)
    {
        // release the resources that this process item had acquired
        // (in practice, this frees the stream for other work)
        gpu_process_item_ptr->release_resources();

        // construct a new CPU process item
        cpu_queue_item_ptr.reset(new cpu_queue_item(gpu_process_item_ptr->batch));

        // push it to the CPU queue
        #pragma omp critical (CPU_Queue_Critical)
        globals.cpu_queue.push(cpu_queue_item_ptr);
    }
}// }}}
#endif // CPU_ONLY

static inline void
#ifdef COUNT
check_cpu_queue (uint64_t &processed_chunks, uint64_t &processed_numbers)
#else // COUNT
check_cpu_queue ()
#endif // COUNT
{// {{{
    std::shared_ptr<cpu_queue_item> cpu_queue_item_ptr;

    #pragma omp critical (CPU_Queue_Critical)
    if (!globals.cpu_queue.empty())
    {
        cpu_queue_item_ptr = globals.cpu_queue.front();
        globals.cpu_queue.pop();
    }

    if (cpu_queue_item_ptr)
    {
        cpu_queue_item_ptr->add_to_box();
        #ifdef COUNT
        ++processed_chunks;
        processed_numbers += cpu_queue_item_ptr->box_indices.size();
        #endif // COUNT
    }
}// }}}

#ifndef CPU_ONLY
static inline bool
check_finish ()
{// {{{
    if (!globals.workers_finished)
        return false;

    bool cpu_queue_empty,
         gpu_batch_queue_empty,
         #ifndef WORKERS_MAKE_BATCHES
         gpu_queue_empty,
         #endif // WORKERS_MAKE_BATCHES
         gpu_process_list_empty;
    
    #ifndef WORKERS_MAKE_BATCHES
    #pragma omp critical (GPU_Queue_Critical)
    gpu_queue_empty = globals.gpu_queue.empty();

    if (!gpu_queue_empty)
        return false;
    #endif // WORKERS_MAKE_BATCHES

    #if defined(MULTI_ROOT) || defined(EXTRA_ROOT_ADD)
    #   pragma omp critical (GPU_Batch_Queue_Critical)
    #endif // MULTI_ROOT, EXTRA_ROOT_ADD
    gpu_batch_queue_empty = globals.gpu_batch_queue.empty();

    if (!gpu_batch_queue_empty)
        return false;

    #if defined(MULTI_ROOT) || defined(EXTRA_ROOT_ADD)
    #   pragma omp critical (GPU_Process_List_Critical)
    #endif // MULTI_ROOT
    gpu_process_list_empty = globals.gpu_process_list.empty();

    if (!gpu_process_list_empty)
        return false;
    
    #pragma omp critical (CPU_Queue_Critical)
    cpu_queue_empty = globals.cpu_queue.empty();

    if (!cpu_queue_empty)
        return false;

    return true;
}// }}}
#endif // CPU_ONLY

#ifndef CPU_ONLY
static void
root_gpu_process ()
{// {{{
    #ifndef NDEBUG
    std::fprintf(stderr, "root_gpu_process : started ...\n");
    #endif // NDEBUG

    #ifdef MULTI_ROOT
    omp_set_num_threads(globals.Nthreads_root_gpu);
    #endif // MULTI_ROOT

    #ifndef NDEBUG
    std::fprintf(stderr, "Using %d threads for root process.\n", globals.Nthreads_root_gpu);
    #endif // NDEBUG

    #ifdef COUNT
    #ifndef EXTRA_ROOT_ADD
    uint64_t processed_numbers = 0UL;
    uint64_t processed_chunks = 0UL;
    #endif // EXTRA_ROOT_ADD
    uint64_t processed_batches = 0UL;
    #endif // COUNT

    #ifdef MULTI_ROOT
    #   ifdef COUNT
    #       ifndef EXTRA_ROOT_ADD
    #           pragma omp parallel \
                           shared(globals) \
                           default(none) \
                           reduction(+:processed_numbers,processed_chunks,processed_batches)
    #       else // EXTRA_ROOT_ADD
    #           pragma omp parallel \
                           shared(globals) \
                           default(none) \
                           reduction(+:processed_batches)
    #       endif // EXTRA_ROOT_ADD
    #   else // COUNT
    #       pragma omp parallel \
                       shared(globals) \
                       default(none)
    #   endif // COUNT
    #endif // MULTI_ROOT
    {
        // we need to keep one temporary across iterations, since a single batch
        // is assembled from a bunch of gpu_queue_items
        std::shared_ptr<gpu_batch_queue_item> gpu_batch_queue_item_ptr { new gpu_batch_queue_item };

        bool can_finish = false;

        // we loop until we know that there's no work left to do
        while (true)
        {
            #ifndef WORKERS_MAKE_BATCHES
            // check if there is an item ready in the GPU queue
            // If yes, append it to the current batch.
            // If that batch is full already, first push it to the gpu_batch_queue
            check_gpu_queue(gpu_batch_queue_item_ptr, can_finish);
            #endif // WORKERS_MAKE_BATCHES

            // check if there is a batch ready to be computed
            #ifdef COUNT
            check_gpu_batch_queue(processed_batches);
            #else // COUNT
            check_gpu_batch_queue();
            #endif // COUNT

            // check if we can find a finished GPU process whose result we can then
            // push to the CPU queue
            check_gpu_process_list();

            // retreive some CPU work if available and perform it
            #ifndef EXTRA_ROOT_ADD
            #ifdef COUNT
            check_cpu_queue(processed_chunks, processed_numbers);
            #else // COUNT
            check_cpu_queue();
            #endif // COUNT
            #endif // EXTRA_ROOT_ADD

            // it is important the the next two are in this order
            // After we realize that all queues are empty and the workers are done,
            // we still need to process the remaining gpu_batch_queue_item
            //
            // Due to this, even after can_finish was already set to true, the queues
            // can fill again. Hence the && check_finish() here.
            // In principle, this shouldn't be an issue since the above statements are
            // in the correct order, but we never know.
            if (can_finish && check_finish())
                break;

            // we look whether all queues are empty and the workers are done
            can_finish = check_finish();
        }
    }

    #ifdef COUNT
    #ifndef EXTRA_ROOT_ADD
    std::fprintf(stderr, "Root processed %lu numbers in %lu CPU-chunks.\n", processed_numbers, processed_chunks);
    #endif // EXTRA_ROOT_ADD
    std::fprintf(stderr, "Root processed %lu GPU-batches.\n", processed_batches);
    #endif // COUNT

    #ifndef NDEBUG
    std::fprintf(stderr, "root_gpu_process : ended.\n");
    #endif // NDEBUG
    
    #ifdef EXTRA_ROOT_ADD
    globals.root_gpu_finished = true;
    #endif // EXTRA_ROOT_ADD
}// }}}
#endif // CPU_ONLY

#if defined(EXTRA_ROOT_ADD) || defined(CPU_ONLY)
static void
root_add_process ()
{// {{{
    #ifndef NDEBUG
    std::fprintf(stderr, "root_add_process : started ...\n");
    #endif // NDEBUG

    #ifdef COUNT
    uint64_t processed_numbers = 0UL;
    uint64_t processed_chunks = 0UL;
    #endif // COUNT

    bool cpu_queue_empty;

    while (true)
    {
        #ifdef COUNT
        check_cpu_queue(processed_chunks, processed_numbers);
        #else // COUNT
        check_cpu_queue();
        #endif // COUNT

        #pragma omp critical (CPU_Queue_Critical)
        cpu_queue_empty = globals.cpu_queue.empty();

        #ifndef CPU_ONLY
        if (cpu_queue_empty && globals.root_gpu_finished)
        #else // CPU_ONLY
        if (cpu_queue_empty && globals.workers_finished)
        #endif // CPU_ONLY
            break;
    }

    #ifdef COUNT
    std::fprintf(stderr, "Root processed %lu numbers in %lu CPU-chunks.\n", processed_numbers, processed_chunks);
    #endif // COUNT

    #ifndef NDEBUG
    std::fprintf(stderr, "root_add_process : ended.\n");
    #endif // NDEBUG
}// }}}
#endif // EXTRA_ROOT_ADD, CPU_ONLY

#endif // ROOT_HPP
