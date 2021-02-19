#ifndef GLOBALS_HPP
#define GLOBALS_HPP

#include "defines.hpp"

#include <string>
#include <omp.h>

#include "queues.hpp"
#include "gpu_handler.hpp"

struct Globals
{
    // gpu handling class
    gpu_handler gpu;

    // communicate to root that workers are finished with all
    // computations
    bool workers_finished = false;

    #ifdef EXTRA_ROOT_ADD
    bool root_gpu_finished = false;
    #endif // EXTRA_ROOT_ADD

    // multithreading environment
    int Nthreads_tot,
        Nthreads_root_gpu,
        Nthreads_root_add,
        Nthreads_workers;

    // function arguments
    uint64_t Nparticles;
    int64_t box_N,
            dim;
    float box_L,
          box_a;
    float *coords,
          *radii,
          *field,
          *box;

    // queues that need to be accessed by all threads
    cpu_queue_t cpu_queue;
    #ifndef WORKERS_MAKE_BATCHES
    gpu_queue_t gpu_queue;
    #endif // WORKERS_MAKE_BATCHES

    // queues that are only accessed by the root thread
    gpu_batch_queue_t  gpu_batch_queue;
    gpu_process_list_t gpu_process_list;

    Globals () = default;

    Globals (uint64_t Nparticles_, int64_t box_N_, int64_t dim_, float box_L_,
             float *coords_, float *radii_, float *field_, float *box_,
             const char *network_file_);
} globals;

// --- Implementation ---

Globals::Globals (uint64_t Nparticles_, int64_t box_N_, int64_t dim_, float box_L_,
                  float *coords_, float *radii_, float *field_, float *box_,
                  const char *network_file_) :
    gpu { std::string(network_file_) },

    Nthreads_tot { omp_get_max_threads() },

    #ifdef MULTI_ROOT
    Nthreads_root_gpu { Nthreads_tot / MULTI_ROOT },
    #else // MULTI_ROOT
    Nthreads_root_gpu { 1 },
    #endif // MULTI_ROOT

    #ifdef EXTRA_ROOT_ADD
    Nthreads_root_add { 1 },
    #else // EXTRA_ROOT_ADD
    Nthreads_root_add { 0 },
    #endif // EXTRA_ROOT_ADD

    #ifdef MULTI_WORKERS
    Nthreads_workers { Nthreads_tot - Nthreads_root_gpu - Nthreads_root_add },
    #else // MULTI_WORKERS
    Nthreads_workers { 1 },
    #endif // MULTI_WORKERS

    Nparticles { Nparticles_ }, box_N { box_N_ }, dim { dim_ }, box_L { box_L_ },
    box_a { box_L / (float)(box_N) },
    coords { coords_ }, radii { radii_ }, field { field_ }, box { box_ }
{ }


#endif // GLOBALS_HPP
