# Introduction

*Voxelize* converts a list of simulation particles which have a field associated with them
into a cubic lattice of voxels.
This can be useful, for example, to measure power spectra and other summary statistics,
as well as to generate data to be used in machine learning applications.

*Voxelize* uses a spherical top-hat kernel for the particle assignments.
Each particle can have an individual value for the radius of this top-hat.
In principle, extension to other kernels should be possible with relatively little modification
of the code; do get in touch if that is would you need.

Mathematically, *Voxelize* computes the field value associated with voxel *i* as

f<sub>i</sub> = &Sigma;<sub>&alpha;</sub> f<sub>&alpha;</sub> V<sub>i,&alpha;</sub> V<sub>voxel</sub><sup>-1</sup>,

where &alpha; indexes the simulation particles, f<sub>&alpha;</sub> is the field associated with them,
V<sub>i,&alpha;</sub> is the overlap volume between voxel *i* and a sphere of radius r<sub>&alpha;</sub>
centered at particle &alpha;'s position, and V<sub>voxel</sub> is the volume of a voxel.
Note that with this definition, intensive fields f<sub>&alpha;</sub> are mapped to intensive fields f<sub>i</sub>,
for example if the input field associated with the particles is the local density,
the output field will be a local density too (and not a mass).

# The two versions of the code

*Voxelize* can be used in two flavours:
* CPU-only;
* CPU+GPU.

The CPU-only version computes the overlap volumes analytically, using a header file provided
by Strobl et al. 2016 (https://github.com/severinstrobl/overlap).
In order to use the CPU-only version, link with `lib/libvoxelize_cpu.a`
and include the header `include/voxelize_cpu.hpp`, which declares the function
```C
void voxelize (uint64_t Nparticles, int64_t box_N, int64_t dim, float box_L,
               float * coords, float * radii, float * field, float * box);
```
Here, `box` is an output buffer that can fit at least `box_N`<sup>3</sup> floats,
`dim` is the dimensionality of the field that is to be gridded, and `coords`
(and `field`, if `dim`>1) are in row-major (C) order.
The code *adds* to `float * box`, without zeroing first. This has the advantage that
you can repeatedly pass the same `box` for different simulation chunks.
Note that all the `float *` inputs will be modified by the code.

The CPU+GPU version achieves higher performance than the CPU-only version by using a neural network
to interpolate the exact overlap volumes (in principle, the network could also be evaluated on a CPU,
but the performance gain over the exact calculation is small in that case).
In order to use the CPU+GPU version, link with `lib/libvoxelize_gpu.a`
and include the header `include/voxelize_gpu.hpp`, which declares the function
```C
void voxelize (..., gpu_handler * gpu);
```
Here, the ellipses stand for the same arguments as in the CPU-only version,
and `gpu_handler` is a class defined in `include/gpu_handler.hpp`.
The only constructor of this class is declared as
```C
gpu_handler::gpu_handler (const std::string &network_dir);
```
Here, `network_dir` should be a directory that contains the files `network.pt` and `Rlims.txt`.
By default, we provide such a directory in `data/network`.
The `gpu_handler` constructor will load the trained network from disk and move it to the GPU(s).
Since this is a somewhat time-consuming process, the code is written in such a way that you
only need to do this once and can then call the `voxelize` function repeatedly with a
pointer to the same `gpu_handler` instance.

We provide a complete example of how to use the code (in both versions) in `src/example.cpp`.


# Performance and how to tune it

For our tests, we used the script `src/example.cpp`. This program loads the gas particles
from an Illustris-type simulation and constructs the gas density field.
The simulation has 256<sup>3</sup> gas particles in a 25 Mpc/*h* box, which we assign
to a 256<sup>3</sup> lattice.

Using one GPU (Tesla P100) together with 10 CPUs, the CPU+GPU version performs the task
in about 3.7 seconds (without construction of the `gpu_handler` and disk I/O).

The CPU-only version takes about 10 times longer, on 10 CPUs. It should be noted that
it is a bit difficult to compare performances in this way, since some users may find it
easier to access a large number of CPUs.

The CPU-only version should require no tuning, and should run fine over a range of thread numbers
(it should be larger than 5 or so for good resource utilization, synchronization overhead may
become relevant with a large number of threads).

We found that the CPU+GPU version performs best with 1 GPU and of the order 8-12 CPU-threads.
Although the code supports multiple GPUs in principle, we found that not much performance could
be gained by running on more than one GPU.
A point that may require tuning is the batch size, which you can find in the `include/defines.hpp`.
In our tests on a GPU with 16GB memory we found 2<sup>17</sup> to be a good choice.
The user may want to adjust this number proportionally to the memory their GPUs have.
If the number of CPU-threads as well as the batch size are chosen well, utilization metrics
shown by e.g. `nvidia-smi` or `gpustat` will exceed 90% during most of the code's runtime.


# Accuracy

Since the CPU+GPU version uses an interpolator, the results will differ from the analytic calculation
which the CPU-only version performs.
Using the script `src/example.cpp`, we compared the outputs of the two algorithms.
A histogram of the relative difference between the two results is shown here:
![](data/networks/network\_accuracy.png)
We see that the CPU+GPU version achieves sub-percent accuracy for the vast majority of voxels,
with most of them off be only a few permille.


# Dependencies


# Before you build the GPU version



# Build
