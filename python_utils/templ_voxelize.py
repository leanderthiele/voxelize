import ctypes as ct
import numpy as np
from numpy.ctypeslib import ndpointer, as_ctypes

# TODO
# the current code does not deal with the mangled names,
# this needs to be implemented
# The names that are to be replaced are marked by << >>

try :
    from importlib import resources as importlib_resources
except ImportError :
    # we're on some ancient python
    import importlib_resources

class Voxelize :

    __cpu_only = <<CPU_ONLY>>
    
    with importlib_resources.path(__name__, "libvoxelize_cpu.so") as fname :
        __libvoxelize_cpu = ct.CDLL(fname);
    if not __cpu_only :
        with importlib_resources.path(__name__, "libvoxelize_gpu.so") as fname :
            __libvoxelize_gpu = ct.CDLL(fname);

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
        __delete_gpu_handler.argtypes = ct.c_void_p

    def __init__(self, use_gpu=False, network_dir=None) :
        if use_gpu :
            if __cpu_only :
                raise RuntimeError('your python wrapper is configured for cpu-only mode.')
            if network_dir is None :
                with importlib_resources.path(__name__, '<<NETWORK_PATH>>') as fname :
                    self.gpu_handler = Voxelize.__new_gpu_handler(fname)
            else :
                self.gpu_handler = Voxelize.__new_gpu_handler(network_dir)
        else :
            if network_dir is not None :
                raise RuntimeError('use_gpu is False but network_dir is specified.')
            self.gpu_handler = None

    def __del__(self) :
        if self.gpu_handler is not None :
            Voxelize.__delete_gpu_handler(self.gpu_handler)

    def __enter__(self) :
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) :
        self.__del__()

    def __call__(self, box_N, box_L, coords, radii, field, box=None) :
        coords = coords.astype(np.float32)
        radii = radii.astype(np.float32)
        field = field.astype(np.float32)
        if box is not None :
            box = box.astype(np.float32)

        assert(len(radii.shape) == 1)
        assert(len(coords.shape) == 2 and coords.shape[1]==3)

        Nparticles = radii.shape[0]
        dim = 1 if len(field.shape) == 1 else field.shape[1]

        assert(coords.shape[0] == Nparticles)
        assert(field.shape[0] == Nparticles)

        if box is None :
            box = np.zeros(box_N*box_N*box_N, dtype=np.float32)

        args = [ Nparticles, box_N, dim, box_L, coords.flatten(), radii, field.flatten() ]
        if self.gpu_handler is not None :
            args.append(self.gpu_handler)
        
        if self.gpu_handler is not None :
            Voxelize.__voxelize_gpu(*args)
        else :
            Voxelize.__voxelize_cpu(*args)

        return box.reshape((box_N, box_N, box_N))
