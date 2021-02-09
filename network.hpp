#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <torch/torch.h>

// TODO implement the network

struct Net : public torch::nn::Module
{
    Net ();

    torch::Tensor forward (torch::Tensor &x);
};

// --- Implementation ---

Net::Net () = default;

torch::Tensor
Net::forward (torch::Tensor &x)
{
    return torch::ones( { x.size(0), 1 }, x.options() );
}

#endif // NETWORK_HPP
