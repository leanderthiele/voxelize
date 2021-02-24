CPU_ONLY = True

import h5py
import numpy as np
from voxelize import Voxelize

fname = '/projects/QUIJOTE/Leander/SU/hydro_test/seed1/0.00000000p/Arepo/snap_004.hdf5'
PartType = 0
box_N = 256

with h5py.File(fname, 'r') as f :
    box_L = f['Header'].attrs['BoxSize']

    # read the required fields
    particles = f['PartType%d'%PartType]
    coordinates = particles['Coordinates'][...]
    density = particles['Density'][...]
    masses = particles['Masses'][...]

radii = np.cbrt(3.0 * masses / density / 4.0 / np.pi)

with Voxelize(use_gpu=not CPU_ONLY) as v :
    box = v(box_L, coordinates, radii, density, box_N)

np.save('box.npy', box)
