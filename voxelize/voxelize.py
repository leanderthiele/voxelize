import ctypes as ct
from os import path
from sys import stdout

import numpy as np
from numpy.ctypeslib import ndpointer, as_ctypes

try :
    from importlib import resources as importlib_resources
except ImportError :
    # we're on some ancient python
    import importlib_resources

def c_str(some_str) :
    if isinstance(some_str, str) :
        return ct.c_char_p(some_str.encode(stdout.encoding))
    elif isinstance(some_str, bytes) :
        return ct.c_char_p(some_str)
    else :
        raise TypeError('not a string or bytes object')

class Voxelize :
    
    with importlib_resources.path('voxelize', 'libvoxelize_cpu.so') as fname :
        __libvoxelize_cpu = ct.CDLL(fname);
    try :
        with importlib_resources.path('voxelize', 'libvoxelize_gpu.so') as fname :
            __libvoxelize_gpu = ct.CDLL(fname);
            __cpu_only = False
    except FileNotFoundError :
        __cpu_only = True
    except OSError as err :
        if 'cannot open shared object file' in err.strerr :
            print('libvoxelize_gpu.so found but library not found : %s'%err.strerror)
            print('Will continue with CPU-only version.')
            __cpu_only = True
        else :
            raise err

    if __cpu_only :
        print('Only the CPU-only flavour of Voxelize is available!')
    else :
        print('Both the CPU-only and the CPU+GPU flavours of Voxelize are available!')

    # for both cpu and gpu
    __common_argtypes = [ ct.c_size_t, # Nparticles
                          ct.c_size_t, # box_N
                          ct.c_size_t, # dim
                          ct.c_float,  # box_L
                          ndpointer(ct.c_float, flags='C_CONTIGUOUS', ndim=1), # coords
                          ndpointer(ct.c_float, flags='C_CONTIGUOUS', ndim=1), # radii
                          ndpointer(ct.c_float, flags='C_CONTIGUOUS', ndim=1), # field
                          ndpointer(ct.c_float, flags='C_CONTIGUOUS', ndim=1), # box
                        ]

    __voxelize_cpu = __libvoxelize_cpu.pyvoxelize
    __voxelize_cpu.restype = None
    __voxelize_cpu.argtypes = __common_argtypes

    if not __cpu_only :
        __voxelize_gpu = __libvoxelize_gpu.pyvoxelize
        __voxelize_gpu.restype = None
        __voxelize_gpu.argtypes = __common_argtypes + [ct.c_void_p, ]

        # gpu handler allocation
        __new_gpu_handler = __libvoxelize_gpu.pynewgpuhandler
        __new_gpu_handler.restype = ct.POINTER(ct.c_int)
            # for some reason ctypes doesn't like to store void pointers, so we do it this way
        __new_gpu_handler.argtypes = [ ct.c_char_p, ]

        # gpu handler free
        __delete_gpu_handler = __libvoxelize_gpu.pydeletegpuhandler
        __delete_gpu_handler.restype = None
        __delete_gpu_handler.argtypes = [ct.c_void_p, ]

    def __init__(self, use_gpu=False, network_dir=None) :
        if use_gpu :
            if Voxelize.__cpu_only :
                raise RuntimeError('your python wrapper is configured for cpu-only mode.')
            if network_dir is None :
                # this is really hacky -- we're assuming that network.pt and Rlims.pt are
                # in the same directory as this script
                network_dir = path.dirname(path.realpath(__file__))
            self.gpu_handler = Voxelize.__new_gpu_handler(c_str(network_dir))
        else :
            if network_dir is not None :
                raise RuntimeError('use_gpu is False but network_dir is specified.')
            self.gpu_handler = None

    def __del__(self) :
        if self.gpu_handler is not None :
            Voxelize.__delete_gpu_handler(self.gpu_handler)
            self.gpu_handler = None

    def __enter__(self) :
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) :
        self.__del__()

    def __call__(self, box_L, coords, radii, field, box) :
        coords = coords.astype(np.float32)
        radii = radii.astype(np.float32)
        field = field.astype(np.float32)

        assert(len(radii.shape) == 1)
        assert(len(coords.shape) == 2 and coords.shape[1]==3)

        Nparticles = radii.shape[0]
        dim = 1 if len(field.shape) == 1 else field.shape[1]

        assert(coords.shape[0] == Nparticles)
        assert(field.shape[0] == Nparticles)

        if isinstance(box, int) :
            box_N = box
            box = np.zeros(box_N*box_N*box_N, dtype=np.float32)
        else :
            assert(len(box.shape) == 3)
            assert(box.shape[0] == box.shape[1] == box.shape[2])
            box_N = box.shape[0]
            box = box.astype(np.float32).flatten()

        args = [ Nparticles, box_N, dim, box_L, coords.flatten(), radii, field.flatten(), box ]
        if self.gpu_handler is not None :
            args.append(self.gpu_handler)
        
        if self.gpu_handler is not None :
            Voxelize.__voxelize_gpu(*args)
        else :
            Voxelize.__voxelize_cpu(*args)

        return box.reshape((box_N, box_N, box_N))
