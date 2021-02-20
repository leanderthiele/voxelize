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
//
// 3) template the whole thing on dimensionality, voxelize_gpu calls the
//    appropriate template
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
         float *coords, float *radii, float *field, float *box,
         #ifndef CPU_ONLY
         gpu_handler * gpu
         #endif // CPU_ONLY
         )
{// {{{
    // initialize the struct that holds all information
    globals = Globals(Nparticles, box_N, dim, box_L, coords,
                      radii, field, box,
                      #ifndef CPU_ONLY
                      gpu
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

#ifdef TESTS
int
main ()
{// {{{
    // the GPU handler needs to be constructed only once and can be used for multiple calls
    // to the voxelize function.
    // This has the advantage that the network is not copied to the GPU every time voxelize()
    // is called.
    #ifndef CPU_ONLY
    const std::string net_fname = "./network_Rmin9.99999978e-03_Rmax1.00000000e+01_.pt";
    auto gpu_ptr = new gpu_handler (net_fname);
    #endif // CPU_ONLY

    // read some hdf5 simulation file
    const std::string fname = "/projects/QUIJOTE/Leander/SU/hydro_test/seed1/0.00000000p/Arepo/snap_004.hdf5";
    const size_t PartType = 0;
    auto fptr = std::make_shared<H5::H5File>(fname, H5F_ACC_RDONLY);
    auto box_L = ReadHDF5::read_header_attr_scalar<double,float>(fptr, "BoxSize");
    auto Nparticles = ReadHDF5::read_header_attr_vector<int32_t,size_t>(fptr, "NumPart_ThisFile", PartType);

    float *coordinates = (float *)ReadHDF5::read_field(fptr, "PartType0/Coordinates", sizeof(float), Nparticles, 3);
    float *density = (float *)ReadHDF5::read_field(fptr, "PartType0/Density", sizeof(float), Nparticles, 1);
    float *masses = (float *)ReadHDF5::read_field(fptr, "PartType0/Masses", sizeof(float), Nparticles, 1);

    // compute the particle radii from their volumes
    float *radii = (float *)std::malloc(Nparticles * sizeof(float));
    for (size_t ii=0; ii != Nparticles; ++ii)
        radii[ii] = std::cbrt(masses[ii] / M_4PI_3f32 / density[ii]);
    
    // allocate an output buffer for the data product and fill with zeros
    int64_t box_N = 256;
    float *box = (float *)std::malloc(box_N * box_N * box_N * sizeof(float));
    for (size_t ii=0; ii != (size_t)(box_N * box_N * box_N); ++ii)
        box[ii] = 0.0F;

    // some debugging stuff
    for (size_t ii=0; ii != Nparticles; ++ii)
    {
        assert(density[ii] >= 0.0F);
        assert(radii[ii] >= 0.0F);
        for (size_t jj=0; jj != 3; ++jj)
            assert(coordinates[ii*3UL+jj]<=box_L && coordinates[ii*3UL+jj]>=0.0F);
    }

    // call the main function
    voxelize(Nparticles, box_N, 1, box_L,
             coordinates, radii, density, box,
             #ifndef CPU_ONLY
             gpu_ptr
             #endif // CPU_ONLY
             );

    // save data product to file
    {
        auto f = std::fopen("box.bin", "wb");
        std::fwrite(box, sizeof(float), box_N*box_N*box_N, f);
        std::fclose(f);
    }

    // clean up
    std::free(box); std::free(coordinates); std::free(density); std::free(masses); std::free(radii);

    #ifndef CPU_ONLY
    delete gpu_ptr;
    #endif // CPU_ONLY

    return 0;
}// }}}
#endif // TESTS
