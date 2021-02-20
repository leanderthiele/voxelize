#include <cstdio>
#include <cassert>
#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <fstream>

#ifdef CPU_ONLY
#   error "train_network.cpp should only be compiled without the CPU_ONLY macro defined."
#endif // CPU_ONLY

#include <torch/torch.h>

#include "network.hpp"

bool gpu_avail;
std::shared_ptr<c10::Device> device_ptr;

static constexpr size_t batchsize = 4096;
static constexpr size_t Nepoch = 3000;
static constexpr size_t Nbatches_epoch = 100;
 
// initial learning rate
static constexpr double learning_rate = 1e-3;
// how many epochs elapse before the learning rate is reduced
static constexpr size_t lr_sched_rate = 300;
// by how much to reduce the learning rate
static constexpr double lr_sched_fact = 0.7;


// only for file naming purposes
static float Rmin = -1.0F;
static float Rmax = -1.0F;

// auxiliary
static constexpr size_t in_stride = 4;
static constexpr size_t out_stride = 1;

// file names
static const std::string network_fname = "network.pt";
static const std::string val_fname = "validation_loss.bin";
static const std::string in_fname = "inputs.bin";
static const std::string out_fname = "outputs.bin";

// how many samples we have
size_t Nsamples = 0;
size_t sample_offsets[4]; // for the 3 batch types, plus one end
size_t sample_lengths[3]; // for the 3 batch types

auto inputs = std::vector<float>();
auto outputs = std::vector<float>();
auto validation_loss = std::vector<float>();

// establishes the GPU environment
void set_device ();

// loads a binary file into a vector
void load_vec (std::vector<float> &vec, const std::string &fname, size_t stride);

// saves a vector to binary file
void save_vec (const std::vector<float> &vec, const std::string &fname);

// establishes the split into training, validation, and testing data
// (fills sample_offsets, sample_lengths)
void split_samples ();

// be careful : they need to be in this order!
enum class BatchType { training, validation, testing };

// gives a new sample, first is the input and second the target
std::pair<torch::Tensor, torch::Tensor> draw_batch (BatchType b);

// computes the loss function
torch::Tensor loss_fct (torch::Tensor &pred, torch::Tensor &targ);

int main ()
{// {{{
    set_device();

    load_vec(inputs, in_fname, in_stride);
    load_vec(outputs, out_fname, out_stride);
    std::fprintf(stderr, "Loaded training data with Rmin=%.8e and Rmax=%.8e\n", Rmin, Rmax);

    split_samples();

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
            auto batch = draw_batch(BatchType::training);
            auto pred = net->forward(batch.first);
            auto loss = loss_fct(batch.second, pred);
            loss.backward();
            optimizer.step();
        }

        // validate
        net->eval();
        auto val_batch = draw_batch(BatchType::validation);
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
    {
        std::ofstream f (network_fname, std::ofstream::binary);
        if (!f.is_open())
            std::fprintf(stderr, "could not open network output file %s\n",
                                 network_fname.c_str());
        f.write((const char *)&Rmin, sizeof Rmin);
        f.write((const char *)&Rmax, sizeof Rmin);
        torch::save(net, f);
        f.close();
    }

    // save validation loss
    save_vec(validation_loss, val_fname);

    // test the network
    net->eval();
    auto test_batch = draw_batch(BatchType::testing);
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
    save_vec(R, "test_R.bin");
    save_vec(targ, "test_targ.bin");
    save_vec(pred, "test_pred.bin");
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

void split_samples ()
{// {{{
    // fill the end indicator
    sample_offsets[3] = Nsamples;

    // we need only a single testing batch
    sample_offsets[(size_t)BatchType::testing] = Nsamples - batchsize;

    // then we want some validation data
    sample_offsets[(size_t)BatchType::validation] = Nsamples - 0.2*Nsamples;

    // the rest is for training
    sample_offsets[(size_t)BatchType::training] = 0;

    // now fill the lengths
    for (size_t ii=0; ii != 3; ++ii)
    {
        sample_lengths[ii] = sample_offsets[ii+1] - sample_offsets[ii];
        assert(sample_lengths[ii] <= Nsamples);
    }
}// }}}

std::pair<torch::Tensor, torch::Tensor> draw_batch (BatchType b)
{// {{{
    static size_t current_idx[] = { 0, 0, 0 };

    static auto opt = torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .requires_grad(true)
                        .device(torch::DeviceType::CPU)
                        .pinned_memory(gpu_avail);

    size_t bidx = (size_t)b;

    torch::Tensor in = torch::empty({batchsize, Net::netw_item_size}, opt);
    torch::Tensor out = torch::empty({batchsize, 1}, opt);

    auto in_acc = in.accessor<float,2>();
    auto out_acc = out.accessor<float,2>();

    for (size_t ii=0; ii != batchsize;
               ++ii, current_idx[bidx]=(current_idx[bidx]+1)%sample_lengths[bidx])
    {
        const float *in_data = inputs.data() + sample_offsets[bidx] + current_idx[bidx]*in_stride;
        // Note : the input that we get from file has the rotations already modded out,
        //        so we don't have to do this here (this saves a bit of runtime)
        Net::input_normalization(in_data, in_acc[ii], /*do_rotations=*/false);

        out_acc[ii][0] = outputs[current_idx[bidx]];
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
