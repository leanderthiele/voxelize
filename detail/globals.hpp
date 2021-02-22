#ifndef GLOBALS_HPP
#define GLOBALS_HPP

#include "defines.hpp"

#include <string>
#include <omp.h>

#ifdef CPU_ONLY
#   include "voxelize_cpu.hpp"
#else // CPU_ONLY
#   include "voxelize_gpu.hpp"
#endif // CPU_ONLY

#include "queues.hpp"

#ifndef CPU_ONLY
#   include "gpu_handler.hpp"
#endif // CPU_ONLY

using namespace Voxelize;

struct Globals
{// {{{
    #ifndef CPU_ONLY
    // gpu handling class
    gpu_handler * gpu;
    #endif // CPU_ONLY

    // communicate to root that workers are finished with all
    // computations
    bool workers_finished = false;

    #if defined(EXTRA_ROOT_ADD) && !defined(CPU_ONLY)
    bool root_gpu_finished = false;
    #endif // EXTRA_ROOT_ADD, CPU_ONLY

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
    #if !defined(WORKERS_MAKE_BATCHES) && !defined(CPU_ONLY)
    gpu_queue_t gpu_queue;
    #endif // WORKERS_MAKE_BATCHES, CPU_ONLY

    #ifndef CPU_ONLY
    // queues that are only accessed by the root thread
    gpu_batch_queue_t  gpu_batch_queue;
    gpu_process_list_t gpu_process_list;
    #endif // CPU_ONLY

    Globals () = default;

    Globals (uint64_t Nparticles_, int64_t box_N_, int64_t dim_, float box_L_,
             float *coords_, float *radii_, float *field_, float *box_
             #ifndef CPU_ONLY
             , gpu_handler * gpu_
             #endif // CPU_ONLY
             );
};// }}}

// --- Implementation ---

Globals::Globals (uint64_t Nparticles_, int64_t box_N_, int64_t dim_, float box_L_,
                  float *coords_, float *radii_, float *field_, float *box_
                  #ifndef CPU_ONLY
                  , gpu_handler * gpu_
                  #endif // CPU_ONLY
                  ) :
    #ifndef CPU_ONLY
    gpu { gpu_ },
    #endif // CPU_ONLY

    Nthreads_tot { omp_get_max_threads() },

    #ifndef CPU_ONLY
    #ifdef MULTI_ROOT
    Nthreads_root_gpu { Nthreads_tot / MULTI_ROOT },
    #else // MULTI_ROOT
    Nthreads_root_gpu { 1 },
    #endif // MULTI_ROOT
    #else // CPU_ONLY
    Nthreads_root_gpu { 0 },
    #endif  // CPU_ONLY

    #if defined(EXTRA_ROOT_ADD) || defined(CPU_ONLY)
    Nthreads_root_add { 1 },
    #else // EXTRA_ROOT_ADD, CPU_ONLY
    Nthreads_root_add { 0 },
    #endif // EXTRA_ROOT_ADD, CPU_ONLY

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
