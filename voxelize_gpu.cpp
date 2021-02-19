#include <cassert>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <omp.h>

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cuda_profiler_api.h"

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
//
// 2) make GPU Handler a stand-alone pointer, which voxelize_gpu takes as
//    an argument. This allows repeated calls of the routine without the need
//    to go through network loading etc every single time.
//
// 3) template the whole thing on dimensionality, voxelize_gpu calls the
//    appropriate template

void
voxelize_gpu(uint64_t Nparticles, int64_t box_N, int64_t dim, float box_L,
             float *coords, float *radii, float *field, float *box,
             const char *network_file)
{
    // initialize the struct that holds all information
    globals = Globals(Nparticles, box_N, dim, box_L, coords,
                      radii, field, box, network_file);

    auto t1 = std::chrono::steady_clock::now();

    #ifndef NDEBUG
    cudaProfilerStart();
    #endif // NDEBUG

    // split into two threads: root and workers
    #ifdef EXTRA_ROOT_ADD
    omp_set_num_threads(3);
    #else // EXTRA_ROOT_ADD
    omp_set_num_threads(2);
    #endif // EXTRA_ROOT_ADD

    // allow nesting
    omp_set_nested(true);

    #pragma omp parallel sections
    {
        #pragma omp section
        root_gpu_process();

        #ifdef EXTRA_ROOT_ADD
        #pragma omp section
        root_add_process();
        #endif // EXTRA_ROOT_ADD

        #pragma omp section
        workers_process();
    }

    #ifndef NDEBUG
    cudaProfilerStop();
    #endif // NDEBUG

    auto t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = t2 - t1;
    std::fprintf(stderr, "voxelize_gpu function took %.4f seconds\n", diff.count());

    #ifdef COUNT
    size_t gpu_process_items = 0;
    for (auto x : globals.gpu_process_list)
        ++gpu_process_items;
    std::fprintf(stderr, "In the end, %lu in gpu_batch_queue, %lu in gpu_process_list, %lu in cpu_queue\n",
                         globals.gpu_batch_queue.size(),
                         gpu_process_items,
                         globals.cpu_queue.size());
    #endif // COUNT
}

#ifdef TESTS
int
main ()
{
    const std::string net_fname = "network.pt";
    const std::string fname = "/projects/QUIJOTE/Leander/SU/hydro_test/seed1/0.00000000p/Arepo/snap_004.hdf5";
    const size_t PartType = 0;
    auto fptr = std::make_shared<H5::H5File>(fname, H5F_ACC_RDONLY);
    auto box_L = ReadHDF5::read_header_attr_scalar<double,float>(fptr, "BoxSize");
    auto Nparticles = ReadHDF5::read_header_attr_vector<int32_t,size_t>(fptr, "NumPart_ThisFile", PartType);

    float *coordinates = (float *)ReadHDF5::read_field(fptr, "PartType0/Coordinates", sizeof(float), Nparticles, 3);
    float *density = (float *)ReadHDF5::read_field(fptr, "PartType0/Density", sizeof(float), Nparticles, 1);
    float *masses = (float *)ReadHDF5::read_field(fptr, "PartType0/Masses", sizeof(float), Nparticles, 1);
    float *radii = (float *)std::malloc(Nparticles * sizeof(float));

    for (size_t ii=0; ii != Nparticles; ++ii)
        radii[ii] = std::cbrt(masses[ii] / M_4PI_3f32 / density[ii]);
    
    int64_t box_N = 256;
    float *box = (float *)std::malloc(box_N * box_N * box_N * sizeof(float));
    for (size_t ii=0; ii != (size_t)(box_N * box_N * box_N); ++ii)
        box[ii] = 0.0F;

    for (size_t ii=0; ii != Nparticles; ++ii)
    {
        assert(density[ii] >= 0.0F);
        assert(radii[ii] >= 0.0F);
        for (size_t jj=0; jj != 3; ++jj)
            assert(coordinates[ii*3UL+jj]<=box_L && coordinates[ii*3UL+jj]>=0.0F);
    }

    voxelize_gpu(Nparticles, box_N, 1, box_L,
                 coordinates, radii, density, box, net_fname.c_str());

    std::free(coordinates); std::free(density); std::free(masses); std::free(radii);

    {
        auto f = std::fopen("box.bin", "wb");
        std::fwrite(box, sizeof(float), box_N*box_N*box_N, f);
        std::fclose(f);
    }

    return 0;
}
#endif // TESTS
