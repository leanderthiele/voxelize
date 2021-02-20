#include <cmath>
#include <string>

#include "H5Cpp.h"

#include "read_hdf5.hpp"

// Include the appropriate header file
#ifdef CPU_ONLY
#   include "voxelize_cpu.hpp"
#else // CPU_ONLY
#   include "voxelize_gpu.hpp"
#endif // CPU_ONLY

#ifndef CPU_ONLY
#ifndef NETWORK_PATH
#   define NETWORK_PATH "./data/network.pt"
#endif // NETWORK_PATH
#endif // CPU_ONLY

// 4pi/3
#ifndef M_4PI_3f32
#   define M_4PI_3f32 4.1887902047863909846168578443726705122628925325001410946332594564F
#endif

int
main ()
{// {{{
    // the GPU handler needs to be constructed only once and can be used for multiple calls
    // to the voxelize function.
    // This has the advantage that the network is not copied to the GPU every time voxelize()
    // is called.
    #ifndef CPU_ONLY
    const std::string net_fname = NETWORK_PATH;
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
             coordinates, radii, density, box
             #ifndef CPU_ONLY
             , gpu_ptr
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
