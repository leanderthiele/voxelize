#include "defines.hpp"

// FIXME
// declare Globals globals before including the headers.
// Then we can make it static and don't need extern anywhere else.
//
// Check that all functions in the header files have static linkage

#include <cassert>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <chrono>
#include <omp.h>

#ifndef CPU_ONLY
#   include "cuda.h"
#   include "cuda_runtime_api.h"
#   include "cuda_profiler_api.h"
#endif // CPU_ONLY

#ifndef NDEBUG
#   include <chrono>
#endif // NDEBUG

#ifdef TESTS
#   include <cstdlib>
#   include <cmath>
#endif // TESTS

// have this before inclusion of the other header
// files so we can have this variable with static linkage
#include "globals.hpp"
static Voxelize::Globals globals;

#ifdef CPU_ONLY
#   include "voxelize_cpu.hpp"
#else // CPU_ONLY
#   include "voxelize_gpu.hpp"
#endif // CPU_ONLY

#include "geometry.hpp"
#include "root.hpp"
#include "workers.hpp"

#ifndef CPU_ONLY
#   include "gpu_handler.hpp"
#endif // CPU_ONLY

namespace Voxelize {

void
voxelize(size_t Nparticles, size_t box_N, size_t dim, float box_L,
         float *coords, float *radii, float *field, float *box
         #ifndef CPU_ONLY
         , gpu_handler * gpu
         #endif // CPU_ONLY
         )
{// {{{
    // initialize the struct that holds all information
    globals = Globals(Nparticles, box_N, dim, box_L, coords,
                      radii, field, box
                      #ifndef CPU_ONLY
                      , gpu
                      #endif // CPU_ONLY
                      );

    #ifndef NDEBUG
    auto t1 = std::chrono::steady_clock::now();
    #endif // NDEBUG

    #if !defined(NDEBUG) && !defined(CPU_ONLY)
    cudaProfilerStart();
    #endif // NDEBUG, CPU_ONLY

    // split into two threads: root and workers
    #if defined(EXTRA_ROOT_ADD) && !defined(CPU_ONLY)
    omp_set_num_threads(3);
    #else // EXTRA_ROOT_ADD, CPU_ONLY
    omp_set_num_threads(2);
    #endif // EXTRA_ROOT_ADD, CPU_ONLY

    // allow nesting
    // -- OMP says it's recommended to use omp_set_max_active_levels() instead,
    //    but this results in much slower code when compiled from the C++ interface.
    //    (for some reason the python wrapper doesn't have a problem with this,
    //     presumably it sets some environment variable internally)
    omp_set_nested(true);

    #pragma omp parallel sections
    {
        #ifndef CPU_ONLY
        #pragma omp section
        root_gpu_process();
        #endif // CPU_ONLY

        #if defined(EXTRA_ROOT_ADD) || defined(CPU_ONLY)
        #pragma omp section
        root_add_process();
        #endif // EXTRA_ROOT_ADD, CPU_ONLY

        #pragma omp section
        workers_process();
    }

    #if !defined(NDEBUG) && !defined(CPU_ONLY)
    cudaProfilerStop();
    #endif // NDEBUG, CPU_ONLY

    #ifndef NDEBUG
    auto t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = t2 - t1;
    std::fprintf(stderr, "voxelize_gpu function took %.4f seconds\n", diff.count());
    #endif // NDEBUG

    #ifdef COUNT
    #ifndef CPU_ONLY
    size_t gpu_process_items = 0;
    for (auto x : globals.gpu_process_list)
        ++gpu_process_items;
    std::fprintf(stderr, "In the end, %lu in gpu_batch_queue, %lu in gpu_process_list, %lu in cpu_queue\n",
                         globals.gpu_batch_queue.size(),
                         gpu_process_items,
                         globals.cpu_queue.size());
    #else // CPU_ONLY
    std::fprintf(stderr, "In the end, %lu in cpu_queue\n",
                         globals.cpu_queue.size());
    #endif // CPU_ONLY
    #endif // COUNT

    // we need to reset the system to its previous state so subsequent calls
    // use the same environment variable
    omp_set_num_threads(globals.Nthreads_tot);
}// }}}


// for the Python wrapper -- it's easier if we have the unmangled names available
extern "C"
{
    void
    pyvoxelize(size_t Nparticles, size_t box_N, size_t dim, float box_L,
               float *coords, float *radii, float *field, float *box
               #ifndef CPU_ONLY
               , gpu_handler * gpu
               #endif // CPU_ONLY
               )
    {
        voxelize(Nparticles, box_N, dim, box_L,
                 coords, radii, field, box
                 #ifndef CPU_ONLY
                 , gpu
                 #endif // CPU_ONLY
                 );
    }
} // extern "C"


} // namespace Voxelize
