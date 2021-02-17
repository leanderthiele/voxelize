#include <cstdio>
#include <cassert>
#include <vector>
#include <string>
#include <utility>
#include <memory>

#include <torch/torch.h>

#include "network.hpp"

static constexpr size_t batchsize = 4096;
static constexpr size_t Nepoch = 100;
static constexpr size_t Nbatches_epoch = 100;

static constexpr double learning_rate = 1e-3;

static constexpr size_t in_stride = 4;
static constexpr size_t out_stride = 1;

static const std::string in_fname = "inputs.bin";
static const std::string out_fname = "outputs.bin";
static const std::string net_fname = "network.pt";
static const std::string val_fname = "validation_loss.bin";

// how many samples we have
size_t Nsamples = 0;
auto inputs = std::vector<float>();
auto outputs = std::vector<float>();
auto validation_loss = std::vector<float>();

// loads a binary file into a vector
void load_vec (std::vector<float> &vec, const std::string &fname, size_t stride);

// gives a new sample, first is the input and second the target
std::pair<torch::Tensor, torch::Tensor> draw_batch ();

// computes the loss function
torch::Tensor loss_fct (torch::Tensor &pred, torch::Tensor &targ);

int main ()
{// {{{
    load_vec(inputs, in_fname, in_stride);
    load_vec(outputs, out_fname, out_stride);

    auto net = std::make_shared<Net>();

    torch::optim::Adam optimizer (net->parameters(), learning_rate);

    for (size_t epoch_idx=0; epoch_idx != Nepoch; ++epoch_idx)
    {
        // update weights
        net->train();
        for (size_t batch_idx=0; batch_idx != Nbatches_epoch; ++batch_idx)
        {
            optimizer.zero_grad();
            auto batch = draw_batch();
            auto pred = net->forward(batch.first);
            auto loss = loss_fct(batch.second, pred);
            loss.backward();
            optimizer.step();
        }

        // validate
        net->eval();
        auto val_batch = draw_batch();
        auto val_pred = net->forward(val_batch.first);
        auto loss = loss_fct(val_batch.second, val_pred);
        validation_loss.push_back(loss.item<float>());

        std::fprintf(stderr, "epoch %lu / %lu\n", epoch_idx+1, Nepoch);
    }

    // save network
    torch::save(net, net_fname);

    // save validation loss
    {
        std::FILE *f = std::fopen(val_fname.c_str(), "wb");
        std::fwrite(validation_loss.data(), sizeof validation_loss[0], validation_loss.size(), f);
        std::fclose(f);
    }
}// }}}

void load_vec (std::vector<float> &vec, const std::string &fname, size_t stride)
{// {{{
    std::FILE *f = std::fopen(fname.c_str(), "rb");
    size_t Nel;

    std::fread(&Nel, sizeof Nel, 1, f);
    std::fprintf(stderr, "Loading %.2f MB from file %s\n", 1e-6 *(float)(Nel*sizeof(float)), fname.c_str());

    if (Nsamples == 0)
        Nsamples = Nel / stride;
    else
        assert(Nsamples == Nel / stride);

    vec.resize(Nel);

    std::fread(vec.data(), sizeof vec[0], Nel, f);

    std::fclose(f);
}// }}}

std::pair<torch::Tensor, torch::Tensor> draw_batch ()
{// {{{
    static size_t current_idx = 0;

    torch::Tensor in = torch::empty({batchsize, Net::netw_item_size}, torch::kFloat32);
    torch::Tensor out = torch::empty({batchsize, 1}, torch::kFloat32);

    auto in_acc = in.accessor<float,2>();
    auto out_acc = out.accessor<float,2>();

    for (size_t ii=0; ii != batchsize; ++ii, current_idx=(current_idx+1)%Nsamples)
    {
        const float *in_data = inputs.data() + current_idx*in_stride;
        Net::input_normalization(in_data, in_acc[ii]);

        out_acc[ii][0] = outputs[current_idx];
    }

    return std::make_pair(in, out);
}// }}}

torch::Tensor loss_fct (torch::Tensor &pred, torch::Tensor &targ)
{// {{{
    static auto MSE = torch::nn::MSELoss();

    return MSE(pred, targ);
}// }}}
