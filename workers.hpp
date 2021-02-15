#ifndef WORKERS_HPP
#define WORKERS_HPP

#include <queue>
#include <memory>
#include <omp.h>

#ifndef NDEBUG
#   include <cstdio>
#endif // NDEBUG

#include "geometry.hpp"
#include "queues.hpp"
#include "queues_implementation.hpp"
#include "globals.hpp"

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

    #ifndef NDEBUG
    uint64_t processed_numbers = 0UL;
    #endif // NDEBUG

    #ifdef MULTI_WORKERS
    #   pragma omp parallel reduction(+:processed_numbers)
    #endif // MULTI_WORKERS
    {// parallel
        std::shared_ptr<cpu_queue_item> cpu_queue_item_ptr { new cpu_queue_item };
        std::shared_ptr<gpu_queue_item> gpu_queue_item_ptr { new gpu_queue_item };

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
        #   pragma omp for schedule(dynamic, 128)
        #endif // MULTI_WORKERS
        for (uint64_t pp=0UL; pp < globals.Nparticles; ++pp)
        {
            float R = globals.radii[pp];
            float *part_centre = globals.coords + 3UL*pp;
            float *weight = globals.field + globals.dim*pp;

            // TODO it's possible that we need to put these in the inner
            //      loops if one particle carries too much data with it
            add_to_cpu_queue_if_full(cpu_queue_item_ptr);
            add_to_gpu_queue_if_full(gpu_queue_item_ptr);

            for (int64_t xx  = (int64_t)(part_centre[0]-R) - 1L;
                         xx <= (int64_t)(part_centre[0]+R);
                       ++xx)
            {
                int64_t idx_x = globals.box_N * globals.box_N
                                * ((globals.box_N+xx%globals.box_N) % globals.box_N);

                for (int64_t yy  = (int64_t)(part_centre[1]-R) - 1L;
                             yy <= (int64_t)(part_centre[1]+R);
                           ++yy)
                {
                    int64_t idx_y = idx_x + globals.box_N
                                            * ((globals.box_N+yy%globals.box_N) % globals.box_N);

                    for (int64_t zz  = (int64_t)(part_centre[2]-R) - 1L;
                                 zz <= (int64_t)(part_centre[2]+R);
                               ++zz)
                    {
                        int64_t idx = idx_y + ((globals.box_N+zz%globals.box_N) % globals.box_N);
                        std::array<float,3> cub { (float)(xx), (float)(yy), (float)(zz) };

                        mod_translations(cub, part_centre);

                        float overlap = 0.0F;

                        auto triviality = is_trivial(cub, R, overlap);

                        if (triviality == trivial_case_e::no_intersect)
                            continue;
                        
                        #ifndef NDEBUG
                        ++processed_numbers;
                        #endif

                        if (triviality == trivial_case_e::non_trivial)
                            // the overlap is not trivially computed
                            gpu_queue_item_ptr->add(idx, cub, R, weight);
                        else
                            // overlap has been trivially computed, needs only to be added
                            cpu_queue_item_ptr->add(idx, weight, overlap);
                    }// for zz
                }// for yy
            }// for xx
        }// for pp

        // clean up if we have any unfinished data in our local buffers
        add_to_cpu_queue_if_full(cpu_queue_item_ptr, true);
        add_to_gpu_queue_if_full(gpu_queue_item_ptr, true);

    }// parallel

    // we are done and should let the root thread know
    globals.workers_finished = true;

    #ifndef NDEBUG
    std::fprintf(stderr, "Workers processed %lu numbers.\n", processed_numbers);
    #endif // NDEBUG

    #ifndef NDEBUG
    std::fprintf(stderr, "workers_process : ended.\n");
    #endif // NDEBUG
}// }}}

#endif // WORKERS_HPP
