// user must define NETWORK_PATH, INPUTS_PATH, OUTPUTS_PATH from compiler

#include <cstdio>
#include <cassert>
#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <fstream>
#include <filesystem>

#include "defines.hpp"

#ifdef CPU_ONLY
#   error "train_network.cpp should only be compiled without the CPU_ONLY macro defined."
#endif // CPU_ONLY

#ifndef NETWORK_PATH
#   error "please define NETWORK_PATH from compiler"
#endif // NETWORK_PATH

#ifndef INPUTS_PATH
#   error "please define INPUTS_PATH from compiler"
#endif // INPUTS_PATH

#ifndef OUTPUTS_PATH
#   error "please define OUTPUTS_PATH from compiler"
#endif // OUTPUTS_PATH

#include <torch/torch.h>

#include "network.hpp"

using namespace Voxelize;

bool gpu_avail;
std::shared_ptr<c10::Device> device_ptr;

static constexpr size_t batchsize = 4096;
static constexpr size_t Nepoch = 4000;
static constexpr size_t Nbatches_epoch = 100;
 
// initial learning rate
static constexpr double learning_rate = 1e-3;
// how many epochs elapse before the learning rate is reduced
static constexpr size_t lr_sched_rate = 400;
// by how much to reduce the learning rate
static constexpr double lr_sched_fact = 0.7;


// only for file naming purposes
static float Rmin = -1.0F;
static float Rmax = -1.0F;

// auxiliary
static constexpr size_t in_stride = 4;
static constexpr size_t out_stride = 1;

// how many samples we have
size_t Nsamples = 0;

#ifdef SPLIT_SAMPLES
size_t sample_offsets[4]; // for the 3 batch types, plus one end
size_t sample_lengths[3]; // for the 3 batch types
#endif // SPLIT_SAMPLES

auto inputs = std::vector<float>();
auto outputs = std::vector<float>();
auto validation_loss = std::vector<float>();

// establishes the GPU environment
void set_device ();

// loads a binary file into a vector
void load_vec (std::vector<float> &vec, const std::string &fname, size_t stride);

// saves a vector to binary file
void save_vec (const std::vector<float> &vec, const std::string &fname);

#ifdef SPLIT_SAMPLES
// establishes the split into training, validation, and testing data
// (fills sample_offsets, sample_lengths)
void split_samples ();
#endif // SPLIT_SAMPLES

#ifdef SPLIT_SAMPLES
// be careful : they need to be in this order!
enum class BatchType { training, validation, testing };
#endif // SPLIT_SAMPLES

// gives a new sample, first is the input and second the target
std::pair<torch::Tensor, torch::Tensor> draw_batch ( 
                                                    #ifdef SPLIT_SAMPLES
                                                    BatchType b
                                                    #endif // SPLIT_SAMPLES
                                                   );

// computes the loss function
torch::Tensor loss_fct (torch::Tensor &pred, torch::Tensor &targ);

// saves the network and some metadata to the directory named network_file
void save_network (std::shared_ptr<Net> net_ptr);

int main ()
{// {{{
    set_device();

    load_vec(inputs, INPUTS_PATH, in_stride);
    load_vec(outputs, OUTPUTS_PATH, out_stride);
    std::fprintf(stderr, "Loaded training data with Rmin=%.8e and Rmax=%.8e\n", Rmin, Rmax);

    #ifdef SPLIT_SAMPLES
    split_samples();
    #endif // SPLIT_SAMPLES

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
            auto batch = draw_batch(
                                    #ifdef SPLIT_SAMPLES
                                    BatchType::training
                                    #endif // SPLIT_SAMPLES
                                   );
            auto pred = net->forward(batch.first);
            auto loss = loss_fct(batch.second, pred);
            loss.backward();
            optimizer.step();
        }

        // validate
        net->eval();
        auto val_batch = draw_batch(
                                    #ifdef SPLIT_SAMPLES
                                    BatchType::validation
                                    #endif // SPLIT_SAMPLES
                                   );
        auto val_pred = net->forward(val_batch.first);
        auto loss = loss_fct(val_batch.second, val_pred);
        validation_loss.push_back(loss.item<float>());

        // adjust learning rate if necessary
        if (!( (epoch_idx+1UL) % lr_sched_rate ))
            for (auto &group : optimizer.param_groups())
                if (group.has_options())
                {
                    auto &options = static_cast<torch::optim::AdamOptions &>(group.options());
                    options.lr(options.lr() * lr_sched_fact);
                }

        std::fprintf(stderr, "epoch %lu / %lu\n", epoch_idx+1, Nepoch);
    }

    // save network
    save_network(net);

    // save validation loss
    save_vec(validation_loss, NETWORK_PATH"/validation_loss.bin");

    // test the network
    net->eval();
    auto test_batch = draw_batch(
                                 #ifdef SPLIT_SAMPLES
                                 BatchType::testing
                                 #endif // SPLIT_SAMPLES
                                );
    auto in_tens = test_batch.first.to(torch::kCPU);
    auto targ_tens = test_batch.second.to(torch::kCPU);
    auto in_acc = in_tens.accessor<float,2>();
    auto targ_acc = targ_tens.accessor<float,2>();
    auto test_pred = net->forward(test_batch.first);
    test_pred = test_pred.to(torch::kCPU);
    std::vector<float> R;
    std::vector<float> targ;
    std::vector<float> pred;
    auto pred_acc = test_pred.accessor<float,2>();
    for (size_t ii=0; ii != batchsize; ++ii)
    {
        R.push_back(in_acc[ii][0]);
        targ.push_back(targ_acc[ii][0]);
        pred.push_back(pred_acc[ii][0]);
    }
    
    // save test results
    save_vec(R, NETWORK_PATH"test_R.bin");
    save_vec(targ, NETWORK_PATH"test_targ.bin");
    save_vec(pred, NETWORK_PATH"test_pred.bin");
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

    float this_Rmin;
    std::fread(&this_Rmin, sizeof this_Rmin, 1, f);
    if (Rmin < 0.0F)
        Rmin = this_Rmin;
    else
        assert(std::fabs(this_Rmin-Rmin) < 1e-5);

    float this_Rmax;
    std::fread(&this_Rmax, sizeof this_Rmax, 1, f);
    if (Rmax < 0.0F)
        Rmax = this_Rmax;
    else
        assert(std::fabs(this_Rmax-Rmax) < 1e-5);

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

void save_vec (const std::vector<float> &vec, const std::string &fname)
{// {{{
    auto f = std::fopen(fname.c_str(), "wb");
    std::fwrite(vec.data(), sizeof vec[0], vec.size(), f);
    std::fclose(f);
}// }}}

#ifdef SPLIT_SAMPLES
void split_samples ()
{// {{{
    // fill the end indicator
    sample_offsets[3] = Nsamples;

    // we need only a single testing batch
    sample_offsets[(size_t)BatchType::testing] = Nsamples - batchsize;

    // then we want some validation data
    sample_offsets[(size_t)BatchType::validation] = 0.8*Nsamples;

    // the rest is for training
    sample_offsets[(size_t)BatchType::training] = 0;

    // now fill the lengths
    for (size_t ii=0; ii != 3; ++ii)
    {
        sample_lengths[ii] = sample_offsets[ii+1] - sample_offsets[ii];
        assert(sample_lengths[ii] <= Nsamples);
    }

    #ifndef NDEBUG
    for (size_t ii=0; ii != 3; ++ii)
    {
        std::fprintf(stderr, "%lu\t%lu\n", sample_offsets[ii], sample_lengths[ii]);
    }
    #endif // NDEBUG
}// }}}
#endif // SPLIT_SAMPLES

std::pair<torch::Tensor, torch::Tensor> draw_batch (
                                                    #ifdef SPLIT_SAMPLES
                                                    BatchType b
                                                    #endif // SPLIT_SAMPLES
                                                   )
{// {{{
    #ifdef SPLIT_SAMPLES
    static size_t current_idx[] = { 0, 0, 0 };
    #else // SPLIT_SAMPLES
    static size_t current_idx = 0;
    #endif // SPLIT_SAMPLES

    static auto opt = torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .requires_grad(false)
                        .device(torch::DeviceType::CPU)
                        .pinned_memory(gpu_avail);

    #ifdef SPLIT_SAMPLES
    size_t bidx = (size_t)b;
    #endif // SPLIT_SAMPLES

    torch::Tensor in = torch::empty({batchsize, Net::netw_item_size}, opt);
    torch::Tensor out = torch::empty({batchsize, 1}, opt);

    auto in_acc = in.accessor<float,2>();
    auto out_acc = out.accessor<float,2>();

    for (size_t ii=0; ii != batchsize; ++ii,
         #ifndef SPLIT_SAMPLES
         current_idx = (current_idx+1) % Nsamples
         #else // SPLIT_SAMPLES
         current_idx[bidx] = (current_idx[bidx]+1) % sample_lengths[bidx]
         #endif // SPLIT_SAMPLES
        )
    {
        const float *in_data = inputs.data()
            #ifndef SPLIT_SAMPLES 
            + current_idx * in_stride;
            #else // SPLIT_SAMPLES
            + (sample_offsets[bidx] + current_idx[bidx])*in_stride;
            #endif // SPLIT_SAMPLES

        // Note : the input that we get from file has the rotations already modded out,
        //        so we don't have to do this here (this saves a bit of runtime)
        Net::input_normalization(in_data, in_acc[ii], /*do_rotations=*/false);

        #ifndef SPLIT_SAMPLES
        out_acc[ii][0] = outputs[current_idx];
        #else // SPLIT_SAMPLES
        out_acc[ii][0] = outputs[ sample_offsets[bidx]+current_idx[bidx] ];
        #endif // SPLIT_SAMPLES
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

void save_network (std::shared_ptr<Net> net_ptr)
{// {{{
    // make sure we have the directory
    std::filesystem::create_directories(NETWORK_PATH);   

    // serialize the network
    torch::save(net_ptr, NETWORK_PATH"/network.pt");

    // write Rmin, Rmax to file
    {
        auto f = std::fopen(NETWORK_PATH"/Rlims.txt", "w");
        std::fprintf(f, "Rmin = %.8e\n", Rmin);
        std::fprintf(f, "Rmax = %.8e\n", Rmax);
        std::fclose(f);
    }

    // write key elements of the network architectue to file
    // (this is just for human readable information purposes)
    {
        auto f = std::fopen(NETWORK_PATH"/architecture.txt", "w");
        std::fprintf(f, "netw_item_size = %lu\n", Net::netw_item_size);
        std::fprintf(f, "Nhidden = %lu\n", Net::Nhidden);
        std::fprintf(f, "Nneurons = %lu\n", Net::Nneurons);
        std::fclose(f);
    }
}// }}}
