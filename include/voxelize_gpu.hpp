#include <cstddef>

#include "gpu_handler.hpp"

namespace Voxelize {

// TODO
// documentation
void
voxelize(size_t Nparticles, size_t box_N, size_t dim, float box_L,
         float *coords, float *radii, float *field, float *box,
         gpu_handler * gpu);

} // namespace Voxelize
