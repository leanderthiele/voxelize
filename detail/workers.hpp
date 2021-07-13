#ifndef WORKERS_HPP
#define WORKERS_HPP

#include "defines.hpp"

#include <queue>
#include <memory>
#include <algorithm>
#include <omp.h>

#ifndef NDEBUG
#   include <cstdio>
#endif // NDEBUG

#ifdef CPU_ONLY
#   include "voxelize_cpu.hpp"
#else // CPU_ONLY
#   include "voxelize_gpu.hpp"
#endif // CPU_ONLY

#include "geometry.hpp"
#include "queues.hpp"
#include "queues_implementation.hpp"
#include "globals.hpp"
#include "overlap_lft_double.hpp"

namespace Voxelize {

// two smaller helper functions (unfortunately we don't seem to be able to template
// them, because I don't know how to pass the names of the critical sections)

static inline void
add_to_cpu_queue_if_full (std::shared_ptr<cpu_queue_item> &cpu_queue_item_ptr,
                          bool can_finish=false)
{// {{{
    if (cpu_queue_item_ptr->is_full()
        || (can_finish && cpu_queue_item_ptr->box_indices.size()))
    {
        #pragma omp critical(CPU_Queue_Critical)
        globals.cpu_queue.push(cpu_queue_item_ptr);
        
        cpu_queue_item_ptr.reset(new cpu_queue_item);
    }
}// }}}

#ifndef CPU_ONLY
#ifndef WORKERS_MAKE_BATCHES
static inline void
add_to_gpu_queue_if_full (std::shared_ptr<gpu_queue_item> &gpu_queue_item_ptr,
                          bool can_finish=false)
{// {{{
    if (gpu_queue_item_ptr->is_full()
        || (can_finish && gpu_queue_item_ptr->box_indices.size()))
    {
        #pragma omp critical(GPU_Queue_Critical)
        globals.gpu_queue.push(gpu_queue_item_ptr);

        gpu_queue_item_ptr.reset(new gpu_queue_item);
    }
}// }}}
#else // WORKERS_MAKE_BATCHES
static inline void
add_to_gpu_batch_queue_if_full (std::shared_ptr<gpu_batch_queue_item> &gpu_batch_queue_item_ptr,
                                bool can_finish=false)
{// {{{
    if (gpu_batch_queue_item_ptr->is_full()
        || (can_finish && gpu_batch_queue_item_ptr->box_indices.size()))
    {
        #pragma omp critical(GPU_Batch_Queue_Critical)
        globals.gpu_batch_queue.push(gpu_batch_queue_item_ptr);

        gpu_batch_queue_item_ptr.reset(new gpu_batch_queue_item);
    }
}// }}}
#endif // WORKERS_MAKE_BATCHES
#endif // CPU_ONLY

static inline float
exact_overlap (const std::array<float,3> &cub, float R)
{// {{{
    const auto cub0 = (Olap::scalar_t)(cub[0]);
    const auto cub1 = (Olap::scalar_t)(cub[1]);
    const auto cub2 = (Olap::scalar_t)(cub[2]);

    const Olap::vector_t v0 {cub0, cub1, cub2};
    const Olap::vector_t v1 {cub0+1.0, cub1, cub2};
    const Olap::vector_t v2 {cub0+1.0, cub1+1.0, cub2};
    const Olap::vector_t v3 {cub0, cub1+1.0, cub2};
    const Olap::vector_t v4 {cub0, cub1, cub2+1.0};
    const Olap::vector_t v5 {cub0+1.0, cub1, cub2+1.0};
    const Olap::vector_t v6 {cub0+1.0, cub1+1.0, cub2+1.0};
    const Olap::vector_t v7 {cub0, cub1+1.0, cub2+1.0};

    const Olap::Hexahedron Hex {v0,v1,v2,v3,v4,v5,v6,v7};

    // unfortunately, in rare cases the overlap code encounters
    // numerical instability.
    // We have no convenient way to check for this in general,
    // but we can discard obviously wrong results.
    // In that case, we try to get around the problem by slightly perturbing
    // the sphere radius.
    for (int ii=0; ii != 100; ++ii)
    {
        // map 0,1,2,3,... -> 0,-1,+1,-2,+2,-3,+3,...
        int jj = (ii/2) * ( (ii&1) * 2 - 1 );

        // get the slightly perturbed radius.
        // NOTE that on the first iteration which in almost all cases will be the
        //      only one jj==0 so no perturbation takes place
        Olap::scalar_t Rpert = R * (1.0 + jj * 1e-8);

        // now compute the overlap volume
        Olap::Sphere Sph ( {0.0, 0.0, 0.0}, Rpert );
        Olap::scalar_t result = Olap::overlap(Sph, Hex);

        // check that the volume is reasonable, return if the case
        // (otherwise we continue perturbing R)
        Olap::scalar_t lo = 0.0;
        Olap::scalar_t hi = std::min(1.0, M_4PI_3 * Rpert * Rpert * Rpert);

        if (result >= lo && result <= hi)
            return result;
    }

    // we have failed to compute something reasonable.
    // We still continue but give the user a warning message
    std::fprintf(stderr, "WARNING failed to compute exact overlap volume for this configuration, setting to zero.\n");
    return 0.0F;
}// }}}

static void
workers_process ()
{// {{{
    #ifndef NDEBUG
    std::fprintf(stderr, "workers_process : started ...\n");
    #endif // NDEBUG

    #ifdef MULTI_WORKERS
    omp_set_num_threads(globals.Nthreads_workers);
    #endif // MULTI_WORKERS

    #ifndef NDEBUG
    std::fprintf(stderr, "workers using %d threads.\n", globals.Nthreads_workers);
    #endif // NDEBUG

    #ifdef COUNT
    uint64_t processed_numbers = 0UL,
             trivial_calculations = 0UL,
             interpolations = 0UL,
             exact_calculations_lo = 0UL,
             exact_calculations_hi = 0UL;
    #endif // COUNT

    #ifdef MULTI_WORKERS
    #   ifdef COUNT
    #       pragma omp parallel reduction(+:processed_numbers,\
                                            trivial_calculations,\
                                            interpolations,\
                                            exact_calculations_lo,\
                                            exact_calculations_hi)
    #   else // COUNT
    #       pragma omp parallel
    #   endif // COUNT
    #endif // MULTI_WORKERS
    {// parallel
        std::shared_ptr<cpu_queue_item> cpu_queue_item_ptr { new cpu_queue_item };

        #ifndef CPU_ONLY
        #ifndef WORKERS_MAKE_BATCHES
        std::shared_ptr<gpu_queue_item> gpu_queue_item_ptr { new gpu_queue_item };
        #else // WORKERS_MAKE_BATCHES
        std::shared_ptr<gpu_batch_queue_item> gpu_batch_queue_item_ptr { new gpu_batch_queue_item };
        #endif // WORKERS_MAKE_BATCHES
        #endif // CPU_ONLY

        // normalize the coordinates
        #ifdef MULTI_WORKERS
        #   pragma omp for schedule(static)
        #endif // MULTI_WORKERS
        for (uint64_t pp=0UL; pp < globals.Nparticles; ++pp)
        {
            globals.radii[pp] /= globals.box_a;
            for (uint64_t dd=0UL; dd < 3UL; ++dd)
                globals.coords[pp*3UL+dd] /= globals.box_a;
        }

        #ifdef MULTI_WORKERS
        #   pragma omp for schedule(dynamic, 16) // NOTE : the chunk size can have a huge effect,
                                                 //        16 is about the best
        #endif // MULTI_WORKERS
        for (uint64_t pp=0UL; pp < globals.Nparticles; ++pp)
        {
            float R = globals.radii[pp];
            float *part_centre = globals.coords + 3UL*pp;
            float *weight = globals.field + globals.dim*pp;

            #ifndef CPU_ONLY
            #ifndef WORKERS_MAKE_BATCHES
            add_to_gpu_queue_if_full(gpu_queue_item_ptr);
            #endif // WORKERS_MAKE_BATCHES
            #endif // CPU_ONLY

            for (int xx  = (int)(part_centre[0]-R) - 1L;
                     xx <= (int)(part_centre[0]+R);
                   ++xx)
            {
                size_t idx_x = globals.box_N * globals.box_N
                               * periodic_idx(xx, globals.box_N);

                for (int yy  = (int)(part_centre[1]-R) - 1L;
                         yy <= (int)(part_centre[1]+R);
                       ++yy)
                {
                    size_t idx_y = idx_x + globals.box_N
                                           * periodic_idx(yy, globals.box_N);

                    for (int zz  = (int)(part_centre[2]-R) - 1L;
                             zz <= (int)(part_centre[2]+R);
                           ++zz)
                    {
                        size_t idx = idx_y + periodic_idx(zz, globals.box_N);
                        std::array<float,3> cub { (float)(xx), (float)(yy), (float)(zz) };

                        mod_translations(cub, part_centre);
                        mod_reflections(cub);

                        float overlap = 0.0F;

                        auto triviality = is_trivial(cub, R, overlap);

                        if (triviality == trivial_case_e::no_intersect)
                            continue;
                        
                        #ifdef COUNT
                        ++processed_numbers;
                        #endif

                        if (triviality == trivial_case_e::non_trivial)
                        // the overlap is not trivially computed
                        {
                            #ifndef CPU_ONLY
                            if (R < globals.gpu->get_Rmin() || R > globals.gpu->get_Rmax())
                            // the radius falls outside the interpolated regime,
                            // we need to do the full analytical calculation
                            {
                            #endif // CPU_ONLY
                                overlap = exact_overlap(cub, R);

                                #ifdef COUNT
                                #ifndef CPU_ONLY
                                if (R < globals.gpu->get_Rmin())
                                    ++exact_calculations_lo;
                                else
                                    ++exact_calculations_hi;
                                #else // CPU_ONLY
                                ++exact_calculations_hi;
                                #endif // CPU_ONLY
                                #endif // COUNT
                            #ifndef CPU_ONLY
                            }
                            else
                            // the radius is in the interpolated regime,
                            // we can use the interpolating network
                            {
                                #ifndef WORKERS_MAKE_BATCHES
                                gpu_queue_item_ptr->add(idx, cub, R, weight);
                                #else // WORKERS_MAKE_BATCHES
                                add_to_gpu_batch_queue_if_full(gpu_batch_queue_item_ptr);
                                gpu_batch_queue_item_ptr->add(idx, cub, R, weight);
                                #endif // WORKERS_MAKE_BATCHES

                                #ifdef COUNT
                                ++interpolations;
                                #endif // COUNT

                                continue;
                            }
                            #endif // CPU_ONLY
                        }
                        #ifdef COUNT
                        else
                            ++trivial_calculations;
                        #endif // COUNT

                        // overlap has been trivially or analytically computed,
                        // needs only to be added
                        add_to_cpu_queue_if_full(cpu_queue_item_ptr);
                        cpu_queue_item_ptr->add(idx, weight, overlap);
                    }// for zz
                }// for yy
            }// for xx
        }// for pp

        // clean up if we have any unfinished data in our local buffers
        add_to_cpu_queue_if_full(cpu_queue_item_ptr, true);

        #ifndef CPU_ONLY
        #ifndef WORKERS_MAKE_BATCHES
        add_to_gpu_queue_if_full(gpu_queue_item_ptr, true);
        #else // WORKERS_MAKE_BATCHES
        add_to_gpu_batch_queue_if_full(gpu_batch_queue_item_ptr, true);
        #endif // WORKERS_MAKE_BATCHES
        #endif // CPU_ONLY

    }// parallel

    // we are done and should let the root thread know
    globals.workers_finished = true;

    #ifdef COUNT
    std::fprintf(stderr, "Workers processed %lu numbers, of which were\n"
                         "\t%.3e exact calculations (R<Rmin),\n"
                         "\t%.3e exact calculations (R>Rmax),\n"
                         "\t%.3e trivial calculations,\n"
                         "\t%.3e interpolations.\n\n",
                         processed_numbers,
                         (double)exact_calculations_lo,
                         (double)exact_calculations_hi,
                         (double)trivial_calculations,
                         (double)interpolations);
    #endif // NDEBUG

    #ifndef NDEBUG
    std::fprintf(stderr, "workers_process : ended.\n");
    #endif // NDEBUG
}// }}}

} // namespace Voxelize

#endif // WORKERS_HPP
