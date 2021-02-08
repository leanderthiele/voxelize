#ifndef GLOBALS_HPP
#define GLOBALS_HPP

#include <omp.h>

#include "queues.hpp"
#include "gpu_handler.hpp"

struct Globals
{
    // multithreading environment
    int Nthreads_tot,
        Nthreads_root,
        Nthreads_workers;

    // gpu handling class
    gpu_handler gpu;

    // queues that need to be accessed by all threads
    cpu_queue_t cpu_queue;
    gpu_queue_t gpu_queue;

    // function arguments
    int64_t Nparticles,
            box_N,
            dim;
    float box_L;
    const float *coords,
                *radii,
                *field;
    float *box;

    Globals (int64_t Nparticles_, int64_t box_N_, int64_t dim_, float box_L_,
             const float *coords_, const float *radii_, const float *field_,
             float *box_);
};

// --- Implementation ---

Globals::Globals (int64_t Nparticles_, int64_t box_N_, int64_t dim_, float box_L_,
                  const float *coords_, const float *radii_, const float *field_,
                  float *box_) :
    Nthreads_tot { omp_get_max_threads() },
    Nthreads_root { 1 },
    Nthreads_workers { Nthreads_tot - Nthreads_root },
    Nparticles { Nparticles_ }, box_N { box_N_ }, dim { dim_ }, box_L { box_L_ },
    coords { coords_ }, radii { radii_ }, field { field_ }, box { box_ }
{ }


#endif // GLOBALS_HPP
