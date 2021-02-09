#include <omp.h>

#include "globals.hpp"
#include "root.hpp"
#include "workers.hpp"


void
voxelize_gpu(uint64_t Nparticles, int64_t box_N, int64_t dim, float box_L,
             float *coords, float *radii, float *field, float *box,
             char *network_file)
{
    // TODO we can remove this later when we have a trained network
    #ifndef NDEBUG
    char no_file_flag[] = "NO_FILE";
    network_file = no_file_flag;
    #endif // NDEBUG

    // initialize the struct that holds all information
    globals = Globals(Nparticles, box_N, dim, box_L, coords, radii, field, box, network_file);

    // split into two threads: root and workers
    omp_set_num_threads(2);

    // allow nesting
    omp_set_nested(true);

    #pragma omp parallel sections
    {
        #pragma omp section
        root_process();

        #pragma omp section
        workers_process();
    }
}

#ifdef TESTS
int
main ()
{
    voxelize_gpu(0, 0, 0, 0.0F, nullptr, nullptr, nullptr, nullptr, nullptr);

    return 0;
}
#endif // TESTS
