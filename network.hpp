#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <cmath>
#include <string>
#include <array>
#include <vector>

#include <torch/torch.h>

#include "geometry.hpp"


struct NetImpl : public torch::nn::Cloneable<NetImpl>
{
    // number of floats in each item in a batch
    static constexpr size_t netw_item_size = 8;

    // number of hidden layers
    static constexpr size_t Nhidden = 4;

    // number of neurons in a layer
    static constexpr size_t Nneurons = 32;

    // fully connected layers
    std::vector<torch::nn::Linear> fc;

    NetImpl ();

    // need to override reset method to make this module cloneable
    void reset () override;

    torch::Tensor forward (torch::Tensor &x);

    template<typename T>
    static void
    input_normalization (std::array<float,3> &cub, float R, T out, bool do_rotations=true);

    template<typename T>
    static void
    // assumes data in the form { R, cub }
    input_normalization (const float *data, T out);

    static void
    input_normalization (std::array<float,3> &cub, float R, std::vector<float> &out, bool do_rotations=true);

private :
    template<typename T>
    static float
    input_normalization_val (const T &cub, float R, size_t idx);
};

TORCH_MODULE(Net);

// --- Implementation ---

NetImpl::NetImpl ()
{// {{{
    reset();
    
    assert(fc.size() == Nhidden+1);
}// }}}

void
NetImpl::reset ()
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
}// }}}

torch::Tensor
NetImpl::forward (torch::Tensor &x)
{// {{{
    // go through the hidden layers
    for (size_t ii=0; ii != Nhidden; ++ii)
        x = torch::leaky_relu(fc[ii]->forward(x));
    
    // apply the final layer with the output activation function
    x = torch::hardsigmoid(fc[Nhidden]->forward(x));
    return x;
}// }}}

template<typename T>
inline float
NetImpl::input_normalization_val (const T &cub, float R, size_t idx)
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
NetImpl::input_normalization (std::array<float,3> &cub, float R, T out, bool do_rotations)
{// {{{
    if (do_rotations)
        mod_rotations(cub);
    for (size_t ii=0; ii != netw_item_size; ++ii)
        out[ii] = input_normalization_val(cub, R, ii);
}// }}}

template<typename T>
inline void
NetImpl::input_normalization (const float *data, T out)
{// {{{
    float R = data[0];
    const float *cub = data+1;
    for (size_t ii=0; ii != netw_item_size; ++ii)
        out[ii] = input_normalization_val(cub, R, ii);
}// }}}

inline void
NetImpl::input_normalization (std::array<float,3> &cub, float R, std::vector<float> &out, bool do_rotations)
{// {{{
    if (do_rotations)
        mod_rotations(cub);
    for (size_t ii=0; ii != netw_item_size; ++ii)
        out.push_back(input_normalization_val(cub, R, ii));
}// }}}

#endif // NETWORK_HPP
