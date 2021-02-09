#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <string>

#include <torch/torch.h>

// TODO implement the network

struct Net : torch::nn::Module
{
    // number of floats in each item in a batch
    static constexpr size_t netw_item_size = 8;

    // number of hidden layers
    static constexpr size_t Nhidden = 4;

    // number of neurons in a layer
    static constexpr size_t Nneurons = 32;

    // fully connected layers
    std::vector<torch::nn::Linear> fc;

    Net ();

    torch::Tensor forward (torch::Tensor &x);
};

// --- Implementation ---

Net::Net () :
    fc { Nhidden+1, torch::nn::Linear{nullptr} }
{
    for (size_t ii=0UL; ii != Nhidden+1UL; ++ii)
        fc[ii] = register_module("fc" + std::to_string(ii),
                                 torch::nn::Linear((ii==0UL)
                                                   ? netw_item_size
                                                   : Nneurons,
                                                   (ii==Nhidden)
                                                   ? 1
                                                   : Nneurons));
    
    assert(fc.size() == Nhidden+1);
}

torch::Tensor
Net::forward (torch::Tensor &x)
{
    // go through the hidden layers
    for (size_t ii=0; ii != Nhidden; ++ii)
        x = torch::leaky_relu(fc[ii]->forward(x));
    
    // apply the final layer with the output activation function
    x = torch::hardsigmoid(fc[Nhidden]->forward(x));
    return x;
}

#endif // NETWORK_HPP
