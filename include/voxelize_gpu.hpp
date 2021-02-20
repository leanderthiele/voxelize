#include <cstdint>

#include "gpu_handler.hpp"

namespace Voxelize {

// TODO
// documentation
void
voxelize(uint64_t Nparticles, int64_t box_N, int64_t dim, float box_L,
         float *coords, float *radii, float *field, float *box,
         gpu_handler * gpu);

} // namespace Voxelize
