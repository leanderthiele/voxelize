module load cudnn/cuda-11.0/8.0.2
module load cudatoolkit/11.0
module load hdf5/gcc/1.10.0

export VOXELIZE=.
export LIBTORCH=${HOME}/pytorch-install

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${LIBTORCH}/lib:${VOXELIZE}/build
