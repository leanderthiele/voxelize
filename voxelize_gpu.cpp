#include <iostream>
#include <memory>
#include <string>
#include <omp.h>

#include "geometry.hpp"
#include "globals.hpp"
#include "root.hpp"
#include "workers.hpp"

#ifdef TESTS
#   include <cstdlib>
#   include <cmath>
#   include "H5Cpp.h"
#   include "read_hdf5.hpp"
#endif // TESTS

void
voxelize_gpu(uint64_t Nparticles, int64_t box_N, int64_t dim, float box_L,
             float *coords, float *radii, float *field, float *box,
             char *network_file)
{
    // TODO we can remove this later when we have a trained network
    char no_file_flag[] = "NO_FILE";
    network_file = no_file_flag;

    // initialize the struct that holds all information
    globals = Globals(Nparticles, box_N, dim, box_L, coords,
                      radii, field, box, network_file);

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
}

#ifdef TESTS
int
main ()
{
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
    
    int64_t box_N = 128;
    float *box = (float *)std::malloc(box_N * box_N * box_N * sizeof(float));

    voxelize_gpu(Nparticles, box_N, 1, box_L, coordinates, radii, density, box, nullptr);

    std::free(coordinates); std::free(density); std::free(masses); std::free(radii);

    return 0;
}
#endif // TESTS
