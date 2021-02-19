#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

// FIXME get rid of the square rooots!!!

#include <cassert>
#include <array>
#include <algorithm>
#include <cmath>

#ifdef TESTS
#   include "overlap_lft_double.hpp"
#   include <random>
#   include <iostream>
#   include <string>
#endif // TESTS

// 1/sqrt(2)
#ifndef M_SQRT1_2f32
#   define M_SQRT1_2f32 0.7071067811865475244008443621048490392848359376884740365883398689F
#endif
#ifndef M_SQRT1_2
#   define M_SQRT1_2 0.7071067811865475244008443621048490392848359376884740365883398689
#endif

// 4pi/3
#ifndef M_4PI_3f32
#   define M_4PI_3f32 4.1887902047863909846168578443726705122628925325001410946332594564F
#endif
#ifndef M_4PI_3
#   define M_4PI_3 4.1887902047863909846168578443726705122628925325001410946332594564
#endif

static inline float
hypotsq (float x, float y, float z)
{// {{{
    return x*x + y*y + z*z;
}// }}}

// Simple functions to mod-out symmetries, 
// need to be called in the order
//    1) translations
//    2) reflections
//    3) rotations
// {{{
static inline void
mod_translations (std::array<float,3> &cub,
                  float sphere_centre[3])
{
    for (int ii=0; ii != 3; ++ii)
        cub[ii] -= sphere_centre[ii];
}

// assumes that translations have already been modded out
static inline void
mod_reflections (std::array<float,3> &cub)
{
    for (auto &x : cub)
        if (x < -0.5F)
            x = - (x + 1.0F);
}

// assumes that translations & reflections have already been modded out
static inline void
mod_rotations (std::array<float,3> &cub)
{
    std::sort(cub.begin(), cub.end());
}
// }}}

// four possible, qualitatively different cases
enum class trivial_case_e { non_trivial, sphere_in_cube, cube_in_sphere, no_intersect };

// helper functions for the if_trivial function below
// {{{
static inline bool
is_sphere_in_cube (const std::array<float,3> &cub, float R)
{
    return R <= 0.5F
           && cub[0] < -R && cub[0]+1.0F > R
           && cub[1] < -R && cub[1]+1.0F > R
           && cub[2] < -R && cub[2]+1.0F > R;
}

static inline bool
is_cube_in_sphere (const std::array<float,3> &cub, float R)
{
    // NOTE : we may be able to optimize this quite a bit by using
    //        quick, exclusive checks first and then only evaluating
    //        some of the hypotsq depending on signs.
    float Rsq = R * R;

    return R >= M_SQRT1_2f32
           && hypotsq(cub[0], cub[1], cub[2]) < Rsq
           && hypotsq(cub[0]+1.0F, cub[1], cub[2]) < Rsq
           && hypotsq(cub[0]+1.0F, cub[1]+1.0F, cub[2]) < Rsq
           && hypotsq(cub[0]+1.0F, cub[1]+1.0F, cub[2]+1.0F) < Rsq
           && hypotsq(cub[0], cub[1]+1.0F, cub[2]) < Rsq
           && hypotsq(cub[0], cub[1]+1.0F, cub[2]+1.0F) < Rsq
           && hypotsq(cub[0], cub[1], cub[2]+1.0F) < Rsq
           && hypotsq(cub[0]+1.0F, cub[1], cub[2]+1.0F) < Rsq;
}

static inline bool
is_no_intersect (const std::array<float,3> &cub, float R)
{
    return hypotsq(std::max(0.0F, cub[0]),
                   std::max(0.0F, cub[1]),
                   std::max(0.0F, cub[2]) ) > R*R;
}
// }}}

// this function checks for trivial cases and returns non-zero if the configuration
// is indeed trivial.
// In that case, the return argument Tvol *vol contains the trivially computable volume.
// Otherwise the return argument's value is undefined.
//
// It is assumed that translations and reflections have already been modded out
// using the previous functions.
static inline trivial_case_e
is_trivial (const std::array<float,3> &cub, float R, float &vol)
{// {{{
    // NOTE : we may be able to optimize this a little bit by using heuristics
    //        on the size of R to order the checks,
    //        but this is probably not going to have much of an effect
    if (is_no_intersect(cub, R))
    {
        return trivial_case_e::no_intersect;
    }
    else if (is_cube_in_sphere(cub, R))
    {
        vol = 1.0F;
        return trivial_case_e::cube_in_sphere;
    }
    else if (is_sphere_in_cube(cub, R))
    {
        vol = M_4PI_3f32 * R * R * R;
        return trivial_case_e::sphere_in_cube;
    }
    else
        return trivial_case_e::non_trivial;
}// }}}

#ifdef TESTS
// {{{
static std::string
trivial_case_to_str (trivial_case_e x)
{
    switch (x)
    {
        case (trivial_case_e::non_trivial) : return "non_trivial";
        case (trivial_case_e::sphere_in_cube) : return "sphere_in_cube";
        case (trivial_case_e::cube_in_sphere) : return "cube_in_sphere";
        case (trivial_case_e::no_intersect) : return "no_intersect";
        default : return "unexpected!";
    }
}

static void
TEST_is_trivial (void)
{
    std::default_random_engine rng (42);
    std::uniform_real_distribution<double> Rdist (0.05, 5.0);
    std::uniform_real_distribution<double> unitdist (-1.0, 1.0);

    constexpr int Nsph = 1000;
    constexpr int Ncub = 1000;

    std::array<int,4> Ncases;
    for (auto &x : Ncases) x = 0;

    for (int ii=0; ii != Nsph; ++ii)
    {
        double R = Rdist(rng);
        Olap::Sphere Sph { {0.0, 0.0, 0.0}, R };

        for (int jj=0; jj != Ncub; ++jj)
        {
            double cub0 = unitdist(rng) * 2.0 * R;
            double cub1 = unitdist(rng) * 2.0 * R;
            double cub2 = unitdist(rng) * 2.0 * R;

            Olap::vector_t v0 {cub0, cub1, cub2};
            Olap::vector_t v1 {cub0+1.0, cub1, cub2};
            Olap::vector_t v2 {cub0+1.0, cub1+1.0, cub2};
            Olap::vector_t v3 {cub0, cub1+1.0, cub2};
            Olap::vector_t v4 {cub0, cub1, cub2+1.0};
            Olap::vector_t v5 {cub0+1.0, cub1, cub2+1.0};
            Olap::vector_t v6 {cub0+1.0, cub1+1.0, cub2+1.0};
            Olap::vector_t v7 {cub0, cub1+1.0, cub2+1.0};

            Olap::Hexahedron Hex {v0,v1,v2,v3,v4,v5,v6,v7};

            double expected_vol = Olap::overlap(Sph, Hex);
            trivial_case_e expected_output
                = (expected_vol < 1e-15) ? trivial_case_e::no_intersect
                  : (std::fabs(expected_vol-1.0) < 1e-15) ? trivial_case_e::cube_in_sphere
                  : (std::fabs(expected_vol - M_4PI_3*R*R*R) < 1e-15) ? trivial_case_e::sphere_in_cube
                  : trivial_case_e::non_trivial;

            ++Ncases[(int)(expected_output)];

            std::array<float,3> cub {(float)cub0, (float)cub1, (float)cub2};
            mod_reflections(cub);

            float vol;
            trivial_case_e got_output = is_trivial(cub, (float)R, vol);
            if (got_output != expected_output)
            {
                std::cout << "*** TEST_is_trivial : not matching." << '\n'
                          << '\t' << "Expected " << trivial_case_to_str(expected_output)
                          << '\t' << "Got " << trivial_case_to_str(got_output)
                          << '\t' << "Overlap volume is " << expected_vol
                          << std::endl;

            }
        }
    }

    // give a summary of the encountered cases
    std::cout << "TEST_is_trivial summary of encountered cases :" << std::endl;
    for (int ii=0; ii != 4; ++ii)
        std::cout << trivial_case_to_str((trivial_case_e)(ii)) << " : "
                  << Ncases[ii] << std::endl;
}// }}}
#endif // TESTS

#endif // GEOMETRY_HPP
