# ================================================================= #
# This Makefile can build the following targets:
#
# 	voxelize_cpu (generates lib/libvoxelize_cpu.a)
# 	voxelize_gpu (generates lib/libvoxelize_gpu.a)
#
# 	voxelize_cpu_shared (generates lib/libvoxelize_cpu.so)
# 	voxelize_gpu_shared (generates lib/libvoxelize_gpu.so)
#
# 	example_cpu
# 	example_gpu
#
# 	python_cpu (runs pip, also generates lib/libvoxelize_cpu.so)
# 	python (runs pip, also generates lib/libvoxelize_cpu.so
# 	                             and lib/libvoxelize_gpu.so)
# 	
# 	generate_samples (generates a script that can be used
# 	                  to generate samples to train the network)
# 	train_network (generates a script to train a new network)
# 	
# ================================================================= #


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
# Change to I (or the equivalent for your compiler) if you don't want to use gcc
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

PIP:= pip
PIPFLAGS:= --user
PYTHON:= python

# directory structure
SRC:= ./src
INCLUDE:= ./include
DETAIL:= ./detail
BUILD:= ./build
LIB:= ./lib
DATA:= ./data
PYUTILS:= ./python_utils
PYVOXELIZE:= ./voxelize

# only required for train_network, example_gpu
# This will also be the network used in the Python package
# if none is supplied
NETWORK_PATH:= $(DATA)/network

# only required for generate_samples, train_network
INPUTS_PATH:= $(DATA)/inputs.bin
OUTPUTS_PATH:= $(DATA)/outputs.bin

CPU_FLAG:= -DCPU_ONLY
GPU_FLAG:= -UCPU_ONLY

CPU_INCL:= -$(EXTERN_INCL) $(EIGEN3)
CPU_LINK:= $(OMPFLAG)

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
                   $(wildcard $(DETAIL)/*.hpp)

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
.PHONY: voxelize_cpu_shared voxelize_gpu_shared
.PHONY: python python_cpu

voxelize_cpu: $(LIB)/libvoxelize_cpu.a lib $(BUILD)/voxelize_cpu.o $(VOXELIZE_CPU_DEP)

voxelize_gpu: $(LIB)/libvoxelize_gpu.a lib $(BUILD)/voxelize_gpu.o $(VOXELIZE_GPU_DEP)

voxelize_cpu_shared: $(LIB)/libvoxelize_cpu.so lib $(BUILD)/voxelize_cpu_fpic.o $(VOXELIZE_CPU_DEP)

voxelize_gpu_shared: $(LIB)/libvoxelize_gpu.so lib $(BUILD)/voxelize_gpu_fpic.o $(VOXELIZE_GPU_DEP)

python:     $(LIB)/libvoxelize_cpu.so lib $(BUILD)/voxelize_cpu_fpic.o $(VOXELIZE_CPU_DEP) \
            $(LIB)/libvoxelize_gpu.so $(BUILD)/voxelize_gpu_fpic.o $(VOXELIZE_GPU_DEP)
	cp $(LIB)/libvoxelize_cpu.so $(PYVOXELIZE)
	cp $(LIB)/libvoxelize_gpu.so $(PYVOXELIZE)
	cp $(NETWORK_PATH)/network.pt $(PYVOXELIZE)
	cp $(NETWORK_PATH)/Rlims.txt $(PYVOXELIZE)
	$(PIP) install -q . $(PIPFLAGS)

python_cpu: $(LIB)/libvoxelize_cpu.so lib $(BUILD)/voxelize_cpu_fpic.o $(VOXELIZE_CPU_DEP)
	cp $(LIB)/libvoxelize_cpu.so $(PYVOXELIZE)
	$(PIP) install -q . $(PIPFLAGS)

$(LIB)/libvoxelize_cpu.a: lib $(BUILD)/voxelize_cpu.o $(VOXELIZE_CPU_DEP)
	$(AR) $(ARFLAGS) $(LIB)/libvoxelize_cpu.a $(BUILD)/voxelize_cpu.o

$(LIB)/libvoxelize_gpu.a: lib $(BUILD)/voxelize_gpu.o $(VOXELIZE_GPU_DEP)
	$(AR) $(ARFLAGS) $(LIB)/libvoxelize_gpu.a $(BUILD)/voxelize_gpu.o

$(LIB)/libvoxelize_cpu.so: lib $(BUILD)/voxelize_cpu_fpic.o $(VOXELIZE_CPU_DEP)
	$(CC) -shared -o $(LIB)/libvoxelize_cpu.so $(BUILD)/voxelize_cpu_fpic.o $(CPU_LINK)

$(LIB)/libvoxelize_gpu.so: lib $(BUILD)/voxelize_gpu_fpic.o $(VOXELIZE_GPU_DEP)
	$(CC) -shared -o $(LIB)/libvoxelize_gpu.so $(BUILD)/voxelize_gpu_fpic.o $(GPU_LINK)

example_cpu: $(LIB)/libvoxelize_cpu.a $(BUILD)/example_cpu.o $(EXAMPLE_CPU_DEP)
	$(CC) -o example_cpu $(BUILD)/example_cpu.o $(CPU_LINK) $(HDF5_LINK) -L$(LIB) -l:libvoxelize_cpu.a

example_gpu: $(LIB)/libvoxelize_gpu.a $(BUILD)/example_gpu.o $(EXAMPLE_GPU_DEP)
	$(CC) -o example_gpu $(BUILD)/example_gpu.o $(GPU_LINK) $(HDF5_LINK) -L$(LIB) -l:libvoxelize_gpu.a

generate_samples: data $(BUILD)/generate_samples.o $(SAMPLES_DEP)
	$(CC) -o generate_samples $(BUILD)/generate_samples.o $(CPU_LINK)

train_network: data $(BUILD)/train_network.o $(TRAIN_DEP)
	$(CC) -o train_network $(BUILD)/train_network.o $(GPU_LINK)


# Object files
$(BUILD)/voxelize_cpu.o: build $(VOXELIZE_CPU_DEP)
	$(CC) -c $(CCFLAGS) $(CPU_FLAG) $(CPU_INCL) $(OMPFLAG) \
              -I$(INCLUDE) -I$(DETAIL) -o $(BUILD)/voxelize_cpu.o $(SRC)/voxelize.cpp

$(BUILD)/voxelize_gpu.o: build $(VOXELIZE_GPU_DEP)
	$(CC) -c $(CCFLAGS) $(GPU_FLAG) $(GPU_INCL) $(OMPFLAG) \
              -I$(INCLUDE) -I$(DETAIL) -o $(BUILD)/voxelize_gpu.o $(SRC)/voxelize.cpp

$(BUILD)/voxelize_cpu_fpic.o: build $(VOXELIZE_CPU_DEP)
	$(CC) -c $(CCFLAGS) -fPIC $(CPU_FLAG) $(CPU_INCL) $(OMPFLAG) \
              -I$(INCLUDE) -I$(DETAIL) -o $(BUILD)/voxelize_cpu_fpic.o $(SRC)/voxelize.cpp

$(BUILD)/voxelize_gpu_fpic.o: build $(VOXELIZE_GPU_DEP)
	$(CC) -c $(CCFLAGS) -fPIC $(GPU_FLAG) $(GPU_INCL) $(OMPFLAG) \
              -I$(INCLUDE) -I$(DETAIL) -o $(BUILD)/voxelize_gpu_fpic.o $(SRC)/voxelize.cpp

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

$(BUILD)/train_network.o: build $(TRAIN_DEP)
	$(CC) -c $(CCFLAGS) $(GPU_FLAG) $(GPU_INCL) \
              -DINPUTS_PATH=\"$(INPUTS_PATH)\" -DOUTPUTS_PATH=\"$(OUTPUTS_PATH)\" \
              -DNETWORK_PATH=\"$(NETWORK_PATH)\" \
              -I$(INCLUDE) -I$(DETAIL) -o $(BUILD)/train_network.o $(SRC)/train_network.cpp

# directories

build:
	mkdir -p $(BUILD)

lib :
	mkdir -p $(LIB)

data:
	mkdir -p $(DATA)

clean :
	rm -rf $(BUILD)
	rm -rf $(LIB)
	rm -f example_cpu
	rm -f example_gpu
	rm -f train_network
	rm -f generate_samples
	rm -f $(PYVOXELIZE)/Rlims.txt
	rm -f $(PYVOXELIZE)/network.pt
	rm -f $(PYVOXELIZE)/libvoxelize_cpu.so
	rm -f $(PYVOXELIZE)/libvoxelize_gpu.so
	$(PIP) uninstall -qy voxelize
