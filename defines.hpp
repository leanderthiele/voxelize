#ifndef DEFINES_HPP
#define DEFINES_HPP

// this is the setting that most likely needs adjustment by the user.
// I found that runtime can be quite sensitive to it, and tuned it.
// The default (2^17) was found to be optimal on a Tesla P100 with 16 GB memory.
// It may be useful to change it proportionally to the GPU memory that
// the user has available.
#define BATCH_SIZE (1 << 17)

// do not use the interpolating neural network, the code will be run
// completely on CPUs.
// No dependence on PyTorch now.
#define CPU_ONLY


// -------------------------------------------
// Other options that are most likely not
// important for the user.

// do not run asserts and get rid of some print statements
// It doesn't make any appreciable difference in terms of runtime
// to leave asserts etc in.
// #define NDEBUG

// produce code that is needed for testing,
// e.g. main in voxelize_gpu.cpp
//  --- currently : has to be on
#define TESTS

// have an extra root thread that only takes
// CPU queue items and adds the data to the box.
// In that case, the other root threads do not add
// to the box
//  --- currently : at least with a single GPU, it saves about a 3rd
#define EXTRA_ROOT_ADD

// the workers write into Torch tensors directly,
// instead of the root threads assembling them
//  --- currently : very much recommended
#define WORKERS_MAKE_BATCHES

// there are multiple root threads talking to the GPU
// The number specifies how many root threads there should be, according to
//      # root threads = # total threads / MULTI_ROOT
//  --- currently : seems like the best option,
//                  with two roots per GPU
#define MULTI_ROOT 3

// give some print-outs during execution, which are useful during debugging
// specifically, it counts how many items go into queues and how many come out
// It doesn't make any appreciable difference in terms of runtime
#define COUNT

// have multiple workers
//  --- currently : very much recommended
#define MULTI_WORKERS

// instead of querying all streams whether they are finished,
// simply pick a random on
//  --- currently : on the canonical 8-2-1-1 setup, it seems to be slightly better
//                  to leave it out
// #define RANDOM_STREAM

// check whether there's enough free memory before pushing a
// new batch to the GPU
//  --- currently : does not block anymore, so better to leave in for safety
#define CHECK_FOR_MEM

// will wrap the tensor copy and forward into a try-catch block,
// which is attempting to catch memory problems.
// The problem here is that the program can freeze instead of
// doing a meaningful about
//  --- currently : not sure!
#define TRY_COMPUTE

#endif // DEFINES_HPP
