#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "defines.hpp"

#include <cmath>
#include <string>
#include <array>
#include <vector>

#ifndef CPU_ONLY
#   include <torch/torch.h>
#endif // CPU_ONLY

#include "geometry.hpp"


struct Net
    #ifndef CPU_ONLY
    : public torch::nn::Cloneable<Net>
    #endif // CPU_ONLY
{// {{{
    // number of floats in each item in a batch
    static constexpr size_t netw_item_size = 8;

    // number of hidden layers
    static constexpr size_t Nhidden = 4;

    // number of neurons in a layer
    static constexpr size_t Nneurons = 32;

    #ifndef CPU_ONLY
    // fully connected layers
    std::vector<torch::nn::Linear> fc;
    #ifdef PRELU
    std::vector<torch::Tensor> prelu_weights;
    #endif // PRELU
    #endif // CPU_ONLY

    #ifndef CPU_ONLY
    Net ();
    #else // CPU_ONLY
    Net () = delete;
    #endif // CPU_ONLY

    #ifndef CPU_ONLY
    // need to override reset method to make this module cloneable
    void reset () override;
    #endif // CPU_ONLY

    #ifndef CPU_ONLY
    torch::Tensor forward (torch::Tensor &x);
    #endif // CPU_ONLY

    template<typename T>
    static void
    input_normalization (std::array<float,3> &cub, float R, T out, bool do_rotations=true);

    template<typename T>
    static void
    // assumes data in the form { R, cub }
    input_normalization (const float *data, T out, bool do_rotations=true);

    static void
    input_normalization (std::array<float,3> &cub, float R, std::vector<float> &out, bool do_rotations=true);

private :
    // T must be such that cub.operator[] is reasonably defined
    template<typename T>
    static float
    input_normalization_val (const T &cub, float R, size_t idx);
};// }}}

// --- Implementation ---

#ifndef CPU_ONLY
Net::Net ()
{// {{{
    reset();
    
    assert(fc.size() == Nhidden+1);
}// }}}
#endif // CPU_ONLY

#ifndef CPU_ONLY
void
Net::reset ()
{// {{{
    fc = std::vector<torch::nn::Linear>(Nhidden+1, torch::nn::Linear{nullptr});
    for (size_t ii=0UL; ii != Nhidden+1UL; ++ii)
        fc[ii] = register_module("fc" + std::to_string(ii),
                                 torch::nn::Linear((ii==0UL)
                                                   ? netw_item_size
                                                   : Nneurons,
                                                   (ii==Nhidden)
                                                   ? 1
                                                   : Nneurons));

    #ifdef PRELU
    prelu_weights = std::vector<torch::Tensor>(Nhidden);
    for (size_t ii=0UL; ii != Nhidden; ++ii)
        prelu_weights[ii] = register_parameter("prelu_weight" + std::to_string(ii),
                                               torch::normal(0.01, 0.003, {Nneurons}));
    #endif // PRELU
}// }}}
#endif // CPU_ONLY

#ifndef CPU_ONLY
torch::Tensor
Net::forward (torch::Tensor &x)
{// {{{
    // go through the hidden layers
    for (size_t ii=0; ii != Nhidden; ++ii)
        #ifdef PRELU
        x = torch::nn::functional::prelu(fc[ii]->forward(x), prelu_weights[ii]);
        #else // PRELU
        x = torch::nn::functional::leaky_relu
            (fc[ii]->forward(x), torch::nn::functional::LeakyReLUFuncOptions().inplace(true));
        #endif // PRELU
    
    // apply the final layer with the output activation function
    x = torch::hardsigmoid(fc[Nhidden]->forward(x));
    return x;
}// }}}
#endif // CPU_ONLY

template<typename T>
inline float
Net::input_normalization_val (const T &cub, float R, size_t idx)
{// {{{
    if (idx == 0UL)
        return R;
    else if (idx < 4UL)
        return cub[idx-1UL];
    else if (idx == 4UL)
        return std::log(R);
    else
        return cub[idx-5UL] / R;
}// }}}

template<typename T>
inline void
Net::input_normalization (std::array<float,3> &cub, float R, T out, bool do_rotations)
{// {{{
    if (do_rotations)
        mod_rotations(cub);
    for (size_t ii=0; ii != netw_item_size; ++ii)
        out[ii] = input_normalization_val(cub, R, ii);
}// }}}

template<typename T>
inline void
Net::input_normalization (const float *data, T out, bool do_rotations)
{// {{{
    float R = data[0];
    std::array<float,3> cub;
    for (size_t ii=0; ii != 3; ++ii)
        cub[ii] = data[ii+1];
    input_normalization(cub, R, out, do_rotations);
}// }}}

inline void
Net::input_normalization (std::array<float,3> &cub, float R, std::vector<float> &out, bool do_rotations)
{// {{{
    if (do_rotations)
        mod_rotations(cub);
    for (size_t ii=0; ii != netw_item_size; ++ii)
        out.push_back(input_normalization_val(cub, R, ii));
}// }}}

#endif // NETWORK_HPP
