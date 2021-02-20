// user must define INPUTS_PATH, OUTPUTS_PATH from compiler

#include <cmath>
#include <cstdio>
#include <vector>
#include <array>
#include <utility>
#include <string>
#include <random>

#include "geometry.hpp"
#include "network.hpp"
#include "overlap_lft_double.hpp"

#ifndef CPU_ONLY
#   error "generate_samples.cpp should only be compiled with the CPU_ONLY macro defined."
#endif // CPU_ONLY

#ifndef INPUTS_PATH
#   error "please define INPUTS_PATH from compiler"
#endif // INPUTS_PATH

#ifndef OUTPUTS_PATH
#   error "please define OUTPUTS_PATH from compiler"
#endif // OUTPUTS_PATH

static constexpr size_t Nsamples = 1 << 26;
static constexpr float  Rmin     = 0.01;
static constexpr float  Rmax     = 10.0;

std::vector<float> inputs;
std::vector<float> outputs;

// file names
static const std::string in_fname = INPUTS_PATH;
static const std::string out_fname = OUTPUTS_PATH;

// draws random input, including the mod_reflections & mod_rotations
std::pair<std::array<float,3>,float> generate_input (); 

// appends input to inputs
void append_input (const std::pair<std::array<float,3>,float> &in);

// computes the output
float generate_output (const std::pair<std::array<float,3>,float> &in);

// appends output to outputs
void append_output (float out);

// saves vector to binary file
// [ the first sizeof(size_t) bytes are the number of elements ]
void save_vec (const std::vector<float> &vec, const std::string &fname);

int main ()
{// {{{
    inputs.reserve(Nsamples * 4UL);
    outputs.reserve(Nsamples);

    for (size_t ii=0; ii != Nsamples; ++ii)
    {
        auto input = generate_input();
        append_input(input);
        auto output = generate_output(input);
        append_output(output);

        if (!((ii+1UL)%10000UL))
            std::fprintf(stderr, "generate_samples : %lu / %lu\n", ii+1UL, Nsamples);
    }

    save_vec(inputs, in_fname);
    save_vec(outputs, out_fname);

    return 0;
}// }}}

// ---- Implementation ----

std::pair<std::array<float,3>,float> generate_input ()
{// {{{
    static std::default_random_engine rng (137);
    static std::uniform_real_distribution<float> logRdist (std::log(Rmin), std::log(Rmax));
    static std::uniform_real_distribution<float> unitdist (-1.0, 1.0);

    float R = std::exp(logRdist(rng));

    std::array<float,3> cub;
    float discard;
    do
    {
        for (size_t ii=0; ii != 3; ++ii)
            cub[ii] = (R+1.0F) * unitdist(rng);

    } while (is_trivial(cub, R, discard)
             != trivial_case_e::non_trivial);

    mod_reflections(cub);
    mod_rotations(cub);

    return std::make_pair(cub, R);
}// }}}

void append_input (const std::pair<std::array<float,3>,float> &in)
{// {{{
    inputs.push_back(in.second);
    for (size_t ii=0; ii != 3; ++ii)
        inputs.push_back(in.first[ii]);
}// }}}

float generate_output (const std::pair<std::array<float,3>,float> &in)
{// {{{
    auto R = in.second;
    
    auto cub0 = (Olap::scalar_t)(in.first[0]);
    auto cub1 = (Olap::scalar_t)(in.first[1]);
    auto cub2 = (Olap::scalar_t)(in.first[2]);

    Olap::Sphere Sph ( {0.0, 0.0, 0.0}, (Olap::scalar_t)R );

    Olap::vector_t v0 {cub0, cub1, cub2};
    Olap::vector_t v1 {cub0+1.0, cub1, cub2};
    Olap::vector_t v2 {cub0+1.0, cub1+1.0, cub2};
    Olap::vector_t v3 {cub0, cub1+1.0, cub2};
    Olap::vector_t v4 {cub0, cub1, cub2+1.0};
    Olap::vector_t v5 {cub0+1.0, cub1, cub2+1.0};
    Olap::vector_t v6 {cub0+1.0, cub1+1.0, cub2+1.0};
    Olap::vector_t v7 {cub0, cub1+1.0, cub2+1.0};

    Olap::Hexahedron Hex {v0,v1,v2,v3,v4,v5,v6,v7};

    float vol_norm = std::min(M_4PI_3f32*R*R*R, 1.0F);

    return Olap::overlap(Sph, Hex) / vol_norm;
}// }}}

void append_output (float out)
{// {{{
    outputs.push_back(out);
}// }}}

void save_vec (const std::vector<float> &vec, const std::string &fname)
{// {{{
    std::FILE *f = std::fopen(fname.c_str(), "wb");
    size_t Nel = vec.size();
    std::fwrite(&Rmin, sizeof Rmin, 1, f);
    std::fwrite(&Rmax, sizeof Rmax, 1, f);
    std::fwrite(&Nel, sizeof Nel, 1, f);
    std::fwrite(vec.data(), sizeof vec[0], Nel, f);
    std::fclose(f);
}// }}}
