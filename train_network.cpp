#include <cstdio>
#include <cassert>
#include <vector>
#include <string>
#include <utility>
#include <memory>

#include <torch/torch.h>

#include "network.hpp"

bool gpu_avail;
std::shared_ptr<c10::Device> device_ptr;

static constexpr size_t batchsize = 4096;
static constexpr size_t Nepoch = 100;
static constexpr size_t Nbatches_epoch = 100;

static constexpr double learning_rate = 1e-3;

// auxiliary
static constexpr size_t in_stride = 4;
static constexpr size_t out_stride = 1;

// file names
static const std::string in_fname = "inputs.bin";
static const std::string out_fname = "outputs.bin";
static const std::string net_fname = "network.pt";
static const std::string val_fname = "validation_loss.bin";

// how many samples we have
size_t Nsamples = 0;
auto inputs = std::vector<float>();
auto outputs = std::vector<float>();
auto validation_loss = std::vector<float>();

// establishes the GPU environment
void set_device ();

// loads a binary file into a vector
void load_vec (std::vector<float> &vec, const std::string &fname, size_t stride);

// gives a new sample, first is the input and second the target
std::pair<torch::Tensor, torch::Tensor> draw_batch ();

// computes the loss function
torch::Tensor loss_fct (torch::Tensor &pred, torch::Tensor &targ);

int main ()
{// {{{
    set_device();

    load_vec(inputs, in_fname, in_stride);
    load_vec(outputs, out_fname, out_stride);

    auto net = std::make_shared<Net>();
    if (gpu_avail)
        net->to(*device_ptr);

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

void set_device ()
{// {{{
    if (torch::cuda::is_available())
    {
        gpu_avail = true;
        device_ptr = std::make_shared<c10::Device>(c10::DeviceType::CUDA);
    }
    else
    {
        gpu_avail = false;
        device_ptr = std::make_shared<c10::Device>(c10::DeviceType::CPU);
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

    static auto opt = torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .requires_grad(true)
                        .device(torch::DeviceType::CPU)
                        .pinned_memory(gpu_avail);

    torch::Tensor in = torch::empty({batchsize, Net::netw_item_size}, opt);
    torch::Tensor out = torch::empty({batchsize, 1}, opt);

    auto in_acc = in.accessor<float,2>();
    auto out_acc = out.accessor<float,2>();

    for (size_t ii=0; ii != batchsize; ++ii, current_idx=(current_idx+1)%Nsamples)
    {
        const float *in_data = inputs.data() + current_idx*in_stride;
        Net::input_normalization(in_data, in_acc[ii]);

        out_acc[ii][0] = outputs[current_idx];
    }

    if (gpu_avail)
    {
        in = in.to(*device_ptr, true);
        out = out.to(*device_ptr, true);
    }

    return std::make_pair(in, out);
}// }}}

torch::Tensor loss_fct (torch::Tensor &pred, torch::Tensor &targ)
{// {{{
    static auto MSE = torch::nn::MSELoss();

    return MSE(pred, targ);
}// }}}
