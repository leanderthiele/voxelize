#ifndef GLOBALS_HPP
#define GLOBALS_HPP

#include <string>
#include <omp.h>

#include "queues.hpp"
#include "gpu_handler.hpp"

struct Globals
{
    // communicate to root that workers are finished with all
    // computations
    bool workers_finished = false;

    // multithreading environment
    int Nthreads_tot,
        Nthreads_root,
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

    // gpu handling class
    gpu_handler gpu;

    // queues that need to be accessed by all threads
    cpu_queue_t cpu_queue;
    gpu_queue_t gpu_queue;

    // queues that are only accessed by the root thread
    gpu_batch_queue_t  gpu_batch_queue;
    gpu_process_list_t gpu_process_list;

    Globals () = default;

    Globals (uint64_t Nparticles_, int64_t box_N_, int64_t dim_, float box_L_,
             float *coords_, float *radii_, float *field_, float *box_,
             char *network_file_);
} globals;

// --- Implementation ---

Globals::Globals (uint64_t Nparticles_, int64_t box_N_, int64_t dim_, float box_L_,
                  float *coords_, float *radii_, float *field_, float *box_,
                  char *network_file_) :
    Nthreads_tot { omp_get_max_threads() },
    Nthreads_root { 1 },
    #ifdef MULTI_WORKERS
    Nthreads_workers { Nthreads_tot - Nthreads_root },
    #else // MULTI_WORKERS
    Nthreads_workers { 1 };
    #endif // MULTI_WORKERS
    Nparticles { Nparticles_ }, box_N { box_N_ }, dim { dim_ }, box_L { box_L_ },
    box_a { box_L / (float)(box_N) },
    coords { coords_ }, radii { radii_ }, field { field_ }, box { box_ },
    gpu { std::string(network_file_) }
{
    #ifndef MULTI_ROOT
    assert(Nthreads_root == 1);
    #endif // MULTI_ROOT
}


#endif // GLOBALS_HPP
