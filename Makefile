# Things the user should need to vary

EIGEN3:= /usr/include/eigen3

TORCH:= /home/lthiele/pytorch-install
CUDA:= /usr/local/cuda-11.0
CUDNN:= /usr/local/cudnn/cuda-11.0/8.0.2

# only required if you want to build src/example.cpp
# (and on most systems not at all if the HDF5 libraries and header files
#  are in standard locations)
HDF5_INCL:=
HDF5_LINK:= -lhdf5 -lhdf5_cpp

# include flag for external headers (to suppress warnings there)
# Change to I if you don't want to use gcc
EXTERN_INCL:= isystem

# you need to edit these if your directory structure is funny
TORCH_INCL:= -$(EXTERN_INCL) $(TORCH)/include -$(EXTERN_INCL) $(TORCH)/include/torch/csrc/api/include/
CUDA_INCL:= -$(EXTERN_INCL) $(CUDA)/include
CUDNN_INCL:= -$(EXTERN_INCL) $(CUDNN)/include

TORCH_LINK:= -L$(TORCH)/lib
CUDA_LINK:= -L$(CUDA)/lib64
CUDNN_LINK:= -L$(CUDNN)/lib64

CC:= g++
CCFLAGS:= -std=c++17 -O3 -g3 -ffast-math -Wall -Wextra -D_GLIBCXX_USE_CXX11_ABI=1
OMPFLAG:= -fopenmp

AR:= ar
ARFLAGS:= rcs

# directory structure
SRC:=./src
INCLUDE:= ./include
DETAIL:= ./detail
BUILD:= ./build
LIB:= ./lib
DATA:= ./data

# only required for train_network, example_gpu
NETWORK_PATH:= $(DATA)/network

# only required for generate_samples, train_network
INPUTS_PATH:= $(DATA)/inputs.bin
OUTPUTS_PATH:= $(DATA)/outputs.bin

CPU_FLAG:= -DCPU_ONLY
GPU_FLAG:= -UCPU_ONLY

CPU_INCL:= -$(EXTERN_INCL) $(EIGEN3)
CPU_LINK:=

GPU_INCL:= $(CPU_INCL) $(TORCH_INCL) $(CUDA_INCL) $(CUDNN_INCL)
GPU_LINK:= $(CPU_LINK) $(TORCH_LINK) $(CUDA_LINK) $(CUDNN_LINK) \
           -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lcudart

# dependencies
VOXELIZE_CPU_DEP:= $(SRC)/voxelize.cpp $(INCLUDE)/voxelize_cpu.hpp \
                   $(INCLUDE)/defines.hpp \
                   $(DETAIL)/geometry.hpp $(DETAIL)/globals.hpp \
                   $(DETAIL)/queues.hpp $(DETAIL)/queues_implementation.hpp \
                   $(DETAIL)/root.hpp $(DETAIL)/workers.hpp  \
                   $(DETAIL)/overlap_lft_double.hpp

VOXELIZE_GPU_DEP:= $(SRC)/voxelize.cpp $(INCLUDE)/voxelize_gpu.hpp \
                   $(INCLUDE)/defines.hpp $(INCLUDE)/gpu_handler.hpp \
                   $(wildcard $(INCLUDE)/*.hpp)

SAMPLE_DEP:= $(SRC)/generate_samples.cpp \
	     $(INCLUDE)/defines.hpp $(DETAIL)/network.hpp \
             $(DETAIL)/geometry.hpp $(DETAIL)/overlap_lft_double.hpp

TRAIN_DEP:= $(SRC)/train_network.cpp \
            $(INCLUDE)/defines.hpp $(DETAIL)/network.hpp $(DETAIL)/geometry.hpp

EXAMPLE_CPU_DEP:= $(SRC)/example.cpp \
                  $(VOXELIZE_CPU_DEP)

EXAMPLE_GPU_DEP:= $(SRC)/example.cpp \
                  $(VOXELIZE_GPU_DEP)

# now the targets

.PHONY: clean
.PHONY: build lib data
.PHONY: voxelize_cpu voxelize_gpu

voxelize_cpu: lib $(BUILD)/voxelize_cpu.o $(VOXELIZE_CPU_DEP)
	$(AR) $(ARFLAGS) $(LIB)/libvoxelize_cpu.a $(BUILD)/voxelize_cpu.o

voxelize_gpu: lib $(BUILD)/voxelize_gpu.o $(VOXELIZE_GPU_DEP)
	$(AR) $(ARFLAGS) $(LIB)/libvoxelize_gpu.a $(BUILD)/voxelize_gpu.o

example_cpu: $(BUILD)/example_cpu.o $(EXAMPLE_CPU_DEP)
	$(CC) -o example_cpu $(BUILD)/example_cpu.o $(CPU_LINK) $(HDF5_LINK) -L$(LIB) -lvoxelize_cpu

example_gpu: $(BUILD)/example_gpu.o $(EXAMPLE_GPU_DEP)
	$(CC) -o example_gpu $(BUILD)/example_gpu.o $(GPU_LINK) $(HDF5_LINK) -L$(LIB) -lvoxelize_gpu

generate_samples: data $(BUILD)/generate_samples.o $(SAMPLES_DEP)
	$(CC) -o generate_samples $(BUILD)/generate_samples.o $(CPU_LINK)

train_network: data $(BUILD)/train_network.o $(TRAIN_DEP)
	$(CC) -o train_network $(BUILD)/train_network.o $(GPU_LINK)


$(BUILD)/voxelize_cpu.o: build $(VOXELIZE_CPU_DEP)
	$(CC) -c $(CCFLAGS) $(CPU_FLAG) $(CPU_INCL) $(OMPFLAG) \
              -I$(INCLUDE) -I$(DETAIL) -o $(BUILD)/voxelize_cpu.o $(SRC)/voxelize.cpp

$(BUILD)/voxelize_gpu.o: build $(VOXELIZE_GPU_DEP)
	$(CC) -c $(CCFLAGS) $(GPU_FLAG) $(GPU_INCL) $(OMPFLAG) \
              -I$(INCLUDE) -I$(DETAIL) -o $(BUILD)/voxelize_gpu.o $(SRC)/voxelize.cpp

$(BUILD)/example_cpu.o: build $(EXAMPLE_CPU_DEP)
	$(CC) -c $(CCFLAGS) $(CPU_FLAG) $(CPU_INCL) $(HDF5_INCL) \
              -I$(INCLUDE) -o $(BUILD)/example_cpu.o $(SRC)/example.cpp

$(BUILD)/example_gpu.o: build $(EXAMPLE_GPU_DEP)
	$(CC) -c $(CCFLAGS) $(GPU_FLAG) $(GPU_INCL) $(HDF5_INCL) \
              -DNETWORK_PATH=\"$(NETWORK_PATH)\" \
              -I$(INCLUDE) -o $(BUILD)/example_gpu.o $(SRC)/example.cpp

$(BUILD)/generate_samples.o: build $(SAMPLE_DEP)
	$(CC) -c $(CCFLAGS) $(CPU_FLAG) $(CPU_INCL) \
              -DINPUTS_PATH=\"$(INPUTS_PATH)\" -DOUTPUTS_PATH=\"$(OUTPUTS_PATH)\" \
              -I$(INCLUDE) -I$(DETAIL) -o $(BUILD)/generate_samples.o $(SRC)/generate_samples.cpp

$(BUILD)/tain_network.o: build $(TRAIN_DEP)
	$(CC) -c $(CCFLAGS) $(GPU_FLAG) $(GPU_INCL) \
              -DINPUTS_PATH=\"$(INPUTS_PATH)\" -DOUTPUTS_PATH=\"$(OUTPUTS_PATH)\" \
              -DNETWORK_PATH=\"$(NETWORK_PATH)\" \
              -I$(INCLUDE) -I$(DETAIL) -o $(BUILD)/train_network.o $(SRC)/train_network.cpp

build:
	mkdir -p $(BUILD)

lib :
	mkdir -p $(LIB)

data:
	mkdir -p $(DATA)

clean :
	rm -r $(BUILD)
	rm -r $(LIB)
