#include "defines.hpp"

#include <cassert>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <omp.h>

#ifndef CPU_ONLY
#   include "cuda.h"
#   include "cuda_runtime_api.h"
#   include "cuda_profiler_api.h"
#endif // CPU_ONLY

#include "geometry.hpp"
#include "globals.hpp"
#include "root.hpp"
#include "workers.hpp"

#ifndef CPU_ONLY
#   include "gpu_handler.hpp"
#endif // CPU_ONLY

#ifdef TESTS
#   include <cstdlib>
#   include <cmath>
#   include <chrono>
#   include "H5Cpp.h"
#   include "read_hdf5.hpp"
#endif // TESTS

// TODO
//
// 1) store Rmin, Rmax in network file name and retrieve it in GPU Handler.
//    If a particle has a radius falling outside this range, call the slow
//    Olap::overlap routine to do the calculation.
//    For diagnostics, count for how many particles this is the case.
//    DONE
//
// 2) make GPU Handler a stand-alone pointer, which voxelize_gpu takes as
//    an argument. This allows repeated calls of the routine without the need
//    to go through network loading etc every single time.
//    DONE
//
// 3) template the whole thing on dimensionality, voxelize_gpu calls the
//    appropriate template
//    DONE
//
// 4) have CPU_ONLY macro
//    -- with the same number of CPUs, this is probably about 10 times slower
//    DONE
//
// 5) have SYNCHRONIZE macro
//
// 6) replace HYPOT calls
//    DONE

void
voxelize(uint64_t Nparticles, int64_t box_N, int64_t dim, float box_L,
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

    auto t1 = std::chrono::steady_clock::now();

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

    auto t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = t2 - t1;
    std::fprintf(stderr, "voxelize_gpu function took %.4f seconds\n", diff.count());

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
}// }}}

