#ifndef DEFINES_HPP
#define DEFINES_HPP

// do not run asserts and get rid of some print statements
// #define NDEBUG

// produce code that is needed for testing,
// e.g. main in voxelize_gpu.cpp
//  --- currently : has to be on
#define TESTS

// have an extra root thread that only takes
// CPU queue items and adds the data to the box.
// In that case, the other root threads do not add
// to the box
//  --- currently : seems like it doesn't make a big difference
#define EXTRA_ROOT_ADD

// the workers write into Torch tensors directly,
// instead of the root threads assembling them
//  --- currently : very much recommended
#define WORKERS_MAKE_BATCHES

// there are multiple root threads talking to the GPU
//  --- currently : seems like the best option,
//                  with two roots per GPU
#define MULTI_ROOT

// give some print-outs during execution
//  --- currently : useful in debugging, then off
#define COUNT

// have multiple workers
//  --- currently : very much recommended
#define MULTI_WORKERS

// instead of querying all streams whether they are finished,
// simply pick a random on
//  --- currently : not sure! (but may be better)
#define RANDOM_STREAM

// check whether there's enough free memory before pushing a
// new batch to the GPU
//  --- currently : not sure!
#define CHECK_FOR_MEM

// will wrap the tensor copy and forward into a try-catch block,
// which is attempting to catch memory problems.
// The problem here is that the program can freeze instead of
// doing a meaningful about
//  --- currently : not sure!
#define TRY_COMPUTE

#endif // DEFINES_HPP
